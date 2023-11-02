#!/usr/bin/env python
# Тут реализована ЛИШЬ ЧАСТИЧНАЯ поддержка SQL — для CREATE TABLE и INSERT выражений
# https://teiid.github.io/teiid-documents/10.2.x/content/reference/BNF_for_SQL_Grammar.html
# https://github.com/orion-orion/TinyPy/blob/main/pytokenizer.py
# https://mathspp.com/blog/lsbasi-apl-part3
# https://habr.com/ru/articles/471862/
# https://wiki.otwartaedukacja.pl/index.php?title=Elementy_j%C4%99zykoznawstwa_dla_programist%C3%B3w
# https://caiorss.github.io/C-Cpp-Notes/embedded_scripting_languages.html
# https://en.wikipedia.org/wiki/Recursive_descent_parser
"""Parse SQL Dumps to JSON Objects"""
import argparse
import binascii

# from collections import defaultdict
import dataclasses
import functools
import io
import itertools
import json
import logging
import os
import re
import string
import sys
import typing
from base64 import b64encode
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Self, Sequence, Type, TypedDict

__author__ = "Sergey M"


class Color(Enum):
    """
    <https://en.wikipedia.org/wiki/ANSI_escape_code>
    <https://stackoverflow.com/a/75985833/2240578>
    """

    CLEAR = RESET = 0
    BLACK = 30
    RED = auto()
    GREEN = auto()
    YELLOW = auto()
    BLUE = auto()
    MAGENTA = auto()
    CYAN = auto()
    WHITE = auto()
    GRAY = 90
    BRIGHT_RED = auto()
    BRIGHT_GREEN = auto()
    BRIGHT_YELLOW = auto()
    BRIGHT_BLUE = auto()
    BRIGHT_MAGENTA = auto()
    BRIGHT_CYAN = auto()
    BRIGHT_WHITE = auto()

    def __str__(self) -> str:
        return f"\033[{self.value}m"


class ColorHandler(logging.StreamHandler):
    COLOR_LEVELS = {
        logging.DEBUG: Color.CYAN,
        logging.INFO: Color.GREEN,
        logging.WARNING: Color.YELLOW,
        logging.ERROR: Color.BRIGHT_RED,
        logging.CRITICAL: Color.RED,
    }

    fmtr = logging.Formatter("[%(levelname).1s]: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        message = self.fmtr.format(record)
        if self.stream.isatty() and (
            col := self.COLOR_LEVELS.get(record.levelno)
        ):
            return f"{col}{message}{Color.RESET}"
        return message


logger = logging.getLogger(__name__)

# Прячем весь вывод (мне не нравится что в тестах выводит что-то)
logger.addHandler(logging.NullHandler())

# При наследовании от просто type:
# TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
# class TokensMeta(EnumType):
#     @property
#     def regexp(cls) -> re.Pattern:
#         return re.compile(
#             "|".join(f"(?P<{tok.name}>{tok.value})" for tok in cls)
#         )


# class Tokens(Enum, metaclass=TokensMeta):
#     WHITE_SPACE = r"\s+"
#     IDENTIFIER = r"[a-zA-Z_]\w+"
#     INT = r"\d+"
#     STRING = r"'([^']|\\')*'"
#     BACKTICK_STRING = r"`([^`]|\\`)*`"
#     MULTILINE_COMMENT = r"/\*[\s\S]*?\*/"
#     INLINE_COMMENT = r"--.*"
#     # Для примитивного парсера не нужны все операторы
#     ...
#     LPAREN = r"\("
#     RPAREN = r"\)"
#     COMMA = ","
#     EQUAL = "="
#     SEMICOLON = ";"
#     EOF = "$"


# К удалению
class AutoName(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[Any],
    ) -> Any:
        return name


# Тут enum.IntFlag бесполезен
class TokenType(AutoName):
    T_ALTER = auto()
    T_AND = auto()
    T_ASSIGN = auto()
    T_BACKTICK_STRING = auto()
    T_BOOLEAN = auto()
    T_COMMA = auto()
    T_CONCAT = auto()
    T_DIV = auto()
    T_DOUBLE_QUOTED_STRING = auto()
    T_EOF = auto()
    T_EQ = auto()
    T_FLOAT = auto()
    T_FROM = auto()
    T_GT = auto()  # greater than
    T_GTE = auto()  # greater than or equal
    T_HEX_STRING = auto()
    T_IDENTIFIER = auto()
    T_INSERT = auto()
    T_INTEGER = auto()
    T_INTO = auto()
    T_LPAREN = auto()
    T_LT = auto()  # less than
    T_LTE = auto()  # less than or equal
    T_MINUS = auto()
    T_MUL = auto()
    T_NE = auto()
    T_NULL = auto()
    T_OR = auto()
    T_PERIOD = auto()
    T_PLUS = auto()
    T_QMARK = auto()
    T_RPAREN = auto()
    T_SELECT = auto()
    T_SEMICOLON = auto()
    T_SET = auto()
    T_STRING = auto()
    T_VALUES = auto()

    # https://www.postgresql.org/docs/current/ddl-constraints.html
    T_CREATE = auto()
    T_TABLE = auto()
    T_IF = auto()
    T_NOT = auto()
    T_EXISTS = auto()
    T_PRIMARY = auto()
    T_FOREIGN = auto()
    T_KEY = auto()
    T_INDEX = auto()
    T_CONSTRAINT = auto()
    T_CHECK = auto()
    T_UNIQUE = auto()
    T_EXCLUDE = auto()
    T_FULLTEXT = auto()
    T_SPATIAL = auto()

    # Всякие операторы которым мне лень имена выдумывать
    T_DUMMY = auto()
    ...

    # Ошибки
    T_UNEXPECTED_CHAR = auto()
    T_INVALID_TOKEN = auto()

    # Костыль
    T_EMPTY = auto()

    def __str__(self) -> str:
        return self.value


class Token(typing.NamedTuple):
    type: TokenType
    value: Any = ""
    lineno: int = -1
    colno: int = -1

    @property
    def repr_value(self) -> Any:
        return (
            "[BINARY]"
            if isinstance(self.value, bytes)
            else repr(
                cut_text(self.value, 20)
                if isinstance(self.value, str)
                else self.value
            )
        )

    def __repr__(self) -> str:
        return f"token {self.repr_value} ({self.type}) at line {self.lineno} and column {self.colno}"


def cut_text(s: str, n: int, dots: str = "…") -> str:
    """
    >>> cut_text('Привет, мир!', 5)
    'Прив…'
    """
    return s[: n - len(dots)] + ["", dots][len(s) > n]


class Error(Exception):
    error_message: str = "An unexpected error has ocurred"

    def __init__(self, error_message: str | None = None):
        self.error_message = error_message or self.error_message
        super().__init__(self.error_message)


# Ошибки токенайзера удалены

# def sort_keys_length(d: dict) -> dict:
#     return dict(sorted(d.items(), key=lambda kv: len(kv[0]), reverse=True))


# Данный токенайзер по быстрому проходиться по файлу, считывая его посимвольно
# Для самых распространенных диалектов работает
class SQLTokenizer:
    ASTERSISK_CHAR = "*"
    BACKTICK_CHAR = "`"
    DOUBLE_QUOTE_CHAR = '"'
    ESCAPE_CHAR = "\\"
    HEX_CHAR = string.hexdigits
    MINUS_CHAR = "-"
    ID_FIRST_CHAR = "@" + string.ascii_letters + "_"
    ID_CHAR = ID_FIRST_CHAR + string.digits
    NEWLINE_CHAR = "\n"
    PERIOD_CHAR = "."
    QUOTE_CHAR = "'"
    SLASH_CHAR = "/"
    SYMBOL_OPERATORS: dict[str, TokenType] = {
        "-": TokenType.T_MINUS,  # substraction
        "-=": TokenType.T_DUMMY,
        ",": TokenType.T_COMMA,
        ";": TokenType.T_SEMICOLON,
        ":=": TokenType.T_ASSIGN,
        "!=": TokenType.T_NE,
        ".": TokenType.T_PERIOD,
        "(": TokenType.T_LPAREN,
        ")": TokenType.T_RPAREN,
        "*": TokenType.T_MUL,
        "/": TokenType.T_DIV,
        "/=": TokenType.T_DUMMY,
        "&": TokenType.T_DUMMY,
        "&=": TokenType.T_DUMMY,
        "%": TokenType.T_DUMMY,
        "%=": TokenType.T_DUMMY,
        "^": TokenType.T_DUMMY,
        "^=": TokenType.T_DUMMY,
        "+": TokenType.T_PLUS,  # addition
        "+=": TokenType.T_DUMMY,
        "<": TokenType.T_LT,
        "<=": TokenType.T_LTE,
        "<>": TokenType.T_NE,
        "=": TokenType.T_EQ,
        ">": TokenType.T_GT,
        ">=": TokenType.T_GTE,
        "|": TokenType.T_DUMMY,
        "|=": TokenType.T_DUMMY,
        "||": TokenType.T_CONCAT,
        # Эти специфичны для SQL-сервер
        "!<": TokenType.T_DUMMY,
        "!>": TokenType.T_DUMMY,
    }

    def __init__(
        self,
        source: io.TextIOBase | str,
        buffer_size: int = -1,  # Учтите, что по дефолту весь файл читается в память
    ) -> None:
        self.char_it = self.read_chars(
            io.StringIO(source) if isinstance(source, str) else source,
            buffer_size,
        )

    # Медленная функция
    def read_chars(self, fp: io.TextIOBase, buffer_size: int) -> Iterable[str]:
        # Может возникнуть ситуация, когда потребуется пропустить N bytes,
        # поэтому `fp.seek(0)`` не нужен
        for c in itertools.chain.from_iterable(
            iter(functools.partial(fp.read, buffer_size), "")
        ):
            if c == self.NEWLINE_CHAR:
                self.lineno += 1
                self.colno = 0
            elif c:
                self.colno += 1
            yield c
        # fix: RuntimeError: generator raised StopIteration
        # возникает если stdin пуст
        yield ""

    def advance(self) -> None:
        self.ch, self.peek_ch = (
            self.peek_ch,
            next(self.char_it, ""),
        )

    def next_token(self) -> Token:
        # выглядит очень коряво, но мы не можем регулярками распарсить файл, тк тогда его нужно будет прочитать весь, а данная утилита нужна для парсинга гигабайтных-дампов
        self.token_lineno = self.lineno
        self.token_colno = self.colno
        self.advance()
        # TODO: вынести в отдельные методы
        # >>> '' in 'abc'
        # True
        if self.ch == "":
            return self.token(TokenType.T_EOF)
        # пропускаем пробельные символы
        if self.ch.isspace():
            while self.peek_ch.isspace():
                self.advance()
            return self.next_token()
        # пропускаем однострочный комментарий
        if self.ch == self.MINUS_CHAR == self.peek_ch:
            while self.ch:
                self.advance()
                if self.ch == self.NEWLINE_CHAR:
                    break
            return self.next_token()
        # пропускаем многострочный комментарий
        if self.ch == self.SLASH_CHAR and self.peek_ch == self.ASTERSISK_CHAR:
            self.advance()
            while self.ch:
                self.advance()
                if (
                    self.ch == self.ASTERSISK_CHAR
                    and self.peek_ch == self.SLASH_CHAR
                ):
                    self.advance()
                    break
            return self.next_token()
        # идентефикаторы, константы, операторы и ключевые слова
        if self.ch in self.ID_FIRST_CHAR:
            val = self.ch
            while self.peek_ch in self.ID_CHAR:
                self.advance()
                val += self.ch
            upper = val.upper()
            match upper:
                case "NULL":
                    return self.token(TokenType.T_NULL, None)
                case "TRUE" | "FALSE":
                    return self.token(TokenType.T_BOOLEAN, val == "TRUE")
                case (
                    "AND"
                    | "CHECK"
                    | "CONSTRAINT"
                    | "CREATE"
                    | "EXCLUDE"
                    | "EXISTS"
                    | "FOREIGN"
                    | "FULLTEXT"
                    | "IF"
                    | "INDEX"
                    | "INSERT"
                    | "INTO"
                    | "KEY"
                    | "NOT"
                    | "OR"
                    | "PRIMARY"
                    | "SPATIAL"
                    | "TABLE"
                    | "UNIQUE"
                    | "VALUES"
                ):
                    return self.token(TokenType[f"T_{upper}"], val)
            return self.token(TokenType.T_IDENTIFIER, val)
        # бинарные данные хранятся в Неведомой Ебанной Хуйне
        if self.ch == "0" and self.peek_ch.lower() == "x":
            val = ""
            self.advance()
            while self.peek_ch in self.HEX_CHAR:
                self.advance()
                val += self.ch.upper()
                # hex-строки могут разбиваться
                if self.peek_ch == self.NEWLINE_CHAR:
                    self.advance()
            if not val:
                # Я пришел к выводу, что кидать ошибки не стоит, так тогда нельзя игнорировать различного рода ошибки
                # raise Error(
                #     f"invalid hex string at line {self.token_lineno} and column {self.token_colno}"
                # )
                return self.token(TokenType.T_INVALID_TOKEN)
            # hex string => bytes
            return self.token(TokenType.T_HEX_STRING, binascii.unhexlify(val))
        # числа
        # self.ch.isnumeric() использовать нельзя:
        # ValueError: invalid literal for int() with base 10: '³'
        if (
            self.ch == self.MINUS_CHAR and self.peek_ch in string.digits
        ) or self.ch in string.digits:
            val = self.ch
            if self.ch == self.MINUS_CHAR:
                self.advance()
                val += self.ch
            while self.peek_ch in string.digits or (
                self.peek_ch == self.PERIOD_CHAR and not self.PERIOD_CHAR in val
            ):
                self.advance()
                val += self.ch
            return (
                self.token(TokenType.T_FLOAT, float(val))
                if self.PERIOD_CHAR in val
                else self.token(TokenType.T_INTEGER, int(val))
            )
        # строки с разными типами кавычек
        for quote_char, token_type in [
            (self.QUOTE_CHAR, TokenType.T_STRING),
            (self.DOUBLE_QUOTE_CHAR, TokenType.T_DOUBLE_QUOTED_STRING),
            (self.BACKTICK_CHAR, TokenType.T_BACKTICK_STRING),
        ]:
            if self.ch == quote_char:
                # кавычки не нужно запоминать
                val = ""
                while True:
                    self.advance()
                    if not self.ch:
                        # Незакрытые кавычки
                        return self.token(TokenType.T_INVALID_TOKEN, val)
                    if self.ch == quote_char:
                        break
                    # https://dev.mysql.com/doc/refman/8.0/en/string-literals.html
                    if self.ch == self.ESCAPE_CHAR:
                        self.advance()
                        match self.ch:
                            case "0":
                                val += "\0"
                            case "b":
                                val += "\b"
                            case "n":
                                val += "\n"
                            case "r":
                                val += "\r"
                            case "t":
                                val += "\t"
                            case "Z":
                                val += chr(26)
                            case "'" | '"' | "\\" | "%" | "_":
                                val += self.ch
                            case _:
                                val += "\\" + self.ch
                    else:
                        val += self.ch
                # нужно преобразовать escape-последовательности
                # нормализуем сначала кавычки, добавив ко всем слеши ("'" и '"')
                # val = re.sub(r'(?<!\\)[\'"`]', r"\\\g<0>", val)
                # return self.token(token_type, ast.literal_eval(f'"{val}"'))
                return self.token(token_type, val)
        # символьные операторы обрабаытваем последними
        op = self.ch
        while self.peek_ch and op + self.peek_ch in self.SYMBOL_OPERATORS:
            self.advance()
            op += self.ch
        try:
            return self.token(self.SYMBOL_OPERATORS[op], op)
        except KeyError as ex:
            return self.token(TokenType.T_UNEXPECTED_CHAR, self.ch)

    def token(self, *args: Any, **kwargs: Any) -> Token:
        return Token(
            *args, **kwargs, colno=self.token_colno, lineno=self.token_lineno
        )

    def tokenize(self) -> Iterable[Token]:
        self.ch = None
        self.colno = 0
        self.lineno = 1
        self.peek_ch = next(self.char_it)
        while tok := self.next_token():
            yield tok
            if tok.type == TokenType.T_EOF:
                break

    __iter__ = tokenize


class ParseError(Error):
    pass


class UnexpectedEnd(ParseError):
    error_message: str = "Unexpected End"


class InsertValues(TypedDict):
    table_name: str
    values: dict[str, Any] | list[Any]


@dataclass
class DumpParser:
    _: dataclasses.KW_ONLY
    ignore_errors: bool = True
    tokenizer_class: Type = SQLTokenizer

    def advance_token(self) -> None:
        self.cur_token, self.next_token = (
            self.next_token,
            # Этот пустой токен тупо костыль, но альтернативу ему я не придумал
            next(self.tokenizer_it, Token(TokenType.T_EMPTY)),
        )

    def peek_token(self, *expected: TokenType) -> bool:
        """Проверить тип следующего токена и сделать его текущим"""
        if self.next_token.type in expected:
            logger.debug("peek %s", self.next_token)
            self.advance_token()
            return True
        return False

    def expect_token(self, *expected: TokenType) -> Self:
        if not self.peek_token(*expected):
            raise ParseError(f"unexpected: {self.next_token}")
        # Когда нечего вернуть, то лучше всего возвращать self
        return self

    # Проритет операторов описан здесь:
    # https://learn.microsoft.com/ru-ru/sql/t-sql/language-elements/operator-precedence-transact-sql?view=sql-server-ver16
    # Чем ниже приоритет тем первее вызывается
    # Этот функционал избыточен, но мне жалко удалять код ниже
    def expr(self) -> Any:
        rv = self.AND()
        while self.peek_token(TokenType.T_OR):
            rv = rv or self.AND()
        return rv

    def AND(self) -> Any:
        rv = self.compare()
        while self.peek_token(TokenType.T_AND):
            rv = rv and self.compare()
        return rv

    def compare(self) -> Any:
        rv = self.addsub()
        while True:
            if self.peek_token(TokenType.T_EQ):
                rv = rv == self.addsub()
            elif self.peek_token(TokenType.T_NE):
                rv = rv != self.addsub()
            elif self.peek_token(TokenType.T_LT):
                rv = rv < self.addsub()
            elif self.peek_token(TokenType.T_LTE):
                rv = rv <= self.addsub()
            elif self.peek_token(TokenType.T_GT):
                rv = rv > self.addsub()
            elif self.peek_token(TokenType.T_GTE):
                rv = rv >= self.addsub()
            else:
                return rv

    def addsub(self) -> Any:
        rv = self.muldiv()
        while True:
            if self.peek_token(TokenType.T_PLUS):
                rv += self.muldiv()
            elif self.peek_token(TokenType.T_MINUS):
                rv -= self.muldiv()
            else:
                return rv

    def muldiv(self) -> Any:
        rv = self.primary()
        while self.peek_token(TokenType.T_MUL, TokenType.T_DIV):
            if self.cur_token.type == TokenType.T_MUL:
                rv *= self.primary()
            else:
                rv /= self.primary()
        return rv

    def primary(self) -> Any:
        # Я не уверен, что он должен быть тут
        if self.peek_token(TokenType.T_NOT):
            return not self.primary()
        if self.peek_token(TokenType.T_LPAREN):
            rv = self.expr()
            self.expect_token(TokenType.T_RPAREN)
            return rv
        if self.peek_token(TokenType.T_IDENTIFIER):
            return None
        return self.expect_token(
            TokenType.T_INTEGER,
            TokenType.T_FLOAT,
            TokenType.T_BOOLEAN,
            TokenType.T_NULL,
            TokenType.T_STRING,
            TokenType.T_DOUBLE_QUOTED_STRING,  # В MySQL можно
            TokenType.T_HEX_STRING,
            TokenType.T_IDENTIFIER,
        ).cur_token.value

    def quoted_identifier(self) -> str:
        return self.expect_token(
            TokenType.T_IDENTIFIER,
            TokenType.T_DOUBLE_QUOTED_STRING,
            TokenType.T_BACKTICK_STRING,
            TokenType.T_HEX_STRING,
        ).cur_token.value

    def table_identifier(self) -> str:
        rv = self.quoted_identifier()
        # table_space.table_name
        if self.peek_token(TokenType.T_PERIOD):
            rv += self.cur_token.value + self.quoted_identifier()
        return rv

    def parse_insert(self) -> Iterable[InsertValues]:
        # INSERT INTO tbl (col1, col2) VALUES ('a', 1);
        logger.debug("parse insert values")
        table_name = self.table_identifier()
        logger.debug(f"{table_name=}")
        # имена колонок опциональны
        column_names = []
        if self.peek_token(TokenType.T_LPAREN):
            while True:
                column_names.append(self.quoted_identifier())
                if self.peek_token(TokenType.T_RPAREN):
                    break
                self.expect_token(TokenType.T_COMMA)
        self.expect_token(TokenType.T_VALUES)
        # counter = 0
        while self.expect_token(TokenType.T_LPAREN):
            values = []
            while True:
                values.append(self.expr())
                if self.peek_token(TokenType.T_RPAREN):
                    break
                self.expect_token(TokenType.T_COMMA)
            column_names = self.table_fields.get(table_name, column_names)
            if column_names:
                if len(column_names) != len(values):
                    raise ParseError(
                        f"missmatch number of column names and number of values for table {table_name!r}"
                    )
                values = dict(zip(column_names, values))
            yield {
                "table_name": table_name,
                "values": values.copy(),
            }
            # counter += 1
            if self.peek_token(TokenType.T_COMMA):
                continue
            self.expect_token(TokenType.T_SEMICOLON)
            return

    def raise_eof(self) -> None:
        if self.peek_token(TokenType.T_EOF):
            raise UnexpectedEnd()

    def parse_create_table(self) -> None:
        # https://www.postgresql.org/docs/current/sql-createtable.html
        # https://dev.mysql.com/doc/refman/8.0/en/create-table.html
        logger.debug("parse create table")
        if self.peek_token(TokenType.T_IF):
            self.expect_token(TokenType.T_NOT)
            self.expect_token(TokenType.T_EXISTS)
        table_name = self.table_identifier()
        logger.debug(f"{table_name=}")
        self.expect_token(TokenType.T_LPAREN)
        # помним, что в дампе может быть несколько таблиц с одинаковым именем
        self.table_fields[table_name] = []
        while True:
            # Я не все скорее всего перечислил ключевые слова для объявления индексов отдельно
            if not self.peek_token(
                TokenType.T_CHECK,
                TokenType.T_CONSTRAINT,
                TokenType.T_FOREIGN,
                TokenType.T_FULLTEXT,
                TokenType.T_INDEX,
                TokenType.T_KEY,
                TokenType.T_PRIMARY,
                TokenType.T_SPATIAL,
                TokenType.T_UNIQUE,
            ):
                column_name = self.quoted_identifier()
                self.table_fields[table_name].append(column_name)
            # Формально синтаксис проверяем
            while not self.peek_token(TokenType.T_COMMA):
                if self.peek_token(TokenType.T_RPAREN):
                    logger.debug(
                        "%r: %r",
                        table_name,
                        self.table_fields[table_name],
                    )
                    return
                # varchar(255) and etc
                if self.peek_token(TokenType.T_LPAREN):
                    while not self.peek_token(TokenType.T_RPAREN):
                        self.raise_eof()
                        self.advance_token()
                else:
                    self.raise_eof()
                    self.advance_token()

    def parse(
        self,
        source: typing.TextIO | str,
        *,
        buffer_size: int = 8192,
        ignore_errors: bool = True,
    ) -> Iterable[InsertValues]:
        logger.debug("reading buffer size: %d bytes", buffer_size)
        # ignore errors можно перезаписать
        ignore_errors = ignore_errors and self.ignore_errors
        logger.debug("ignore parse errors: %s", ["off", "on"][ignore_errors])
        self.tokenizer = self.tokenizer_class(source, buffer_size)
        self.tokenizer_it = iter(self.tokenizer)
        self.next_token = Token(TokenType.T_EMPTY)
        # Сделает текущий пустым
        self.advance_token()
        # Используем list так как важен порядок объявления колонок
        # Я не учел, что можно объявить несколько таблиц с одинаковым именем в разных неймспейсах
        # self.table_fields = defaultdict(list)
        self.table_fields = {}
        while not self.peek_token(TokenType.T_EOF, TokenType.T_EMPTY):
            try:
                if self.peek_token(TokenType.T_CREATE):
                    if self.peek_token(TokenType.T_TABLE):
                        self.parse_create_table()
                elif self.peek_token(TokenType.T_INSERT):
                    if self.peek_token(TokenType.T_INTO):
                        yield from self.parse_insert()
                else:
                    self.advance_token()
            except ParseError as ex:
                if not ignore_errors:
                    raise ex
                logger.warning(ex)

        logger.info("finished")


class NameSpace(argparse.Namespace):
    input: typing.TextIO
    output: typing.TextIO
    debug: bool
    fail_on_error: bool
    buffer_size: int


def _parse_args(argv: Sequence[str] | None) -> NameSpace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        default="-",
        help="input file",
        type=argparse.FileType(errors="ignore"),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="output file",
        type=argparse.FileType("w+"),
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="show debug messages",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-b",
        "--buffer-size",
        "--buffer",
        help="buffer size for reading: <SIZE> [ <UNIT> ]; supported units: k, m, g (case insensitive)",
        default="8K",
        type=str_to_number,
    )
    parser.add_argument(
        "--fail-on-error",
        help="not ignore parse errors",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args(argv, namespace=NameSpace())
    return args


def str_to_number(
    s: str, *, units: typing.Sequence[str] = "KMG", base: int = 1024
) -> int:
    # >>> str_to_number('\t128\n')
    # 128
    # >>> str_to_number('128M')
    # 134217728
    # >>> str_to_number('131,072 Kebabytes')
    # 134217728
    if not (match := re.match(r"\s*(\d[,\d]*)\s*([a-zA-Z]?)", s)):
        raise ValueError(f"invalid size: {s!r}")
    size, unit = match.groups()
    return int(size.replace(",", "")) * (
        base ** -~units.lower().index(unit.lower()) if unit else 1
    )


# def str2bool(v: str) -> bool:
#     return v.lower() in ("yes", "true", "t", "1", "on")


def main(argv: Sequence[str] | None = None) -> int | None:
    args = _parse_args(argv)
    # logger тормозная вещь так как блокирует основной поток
    logger.addHandler(ColorHandler())
    if args.debug:
        logger.setLevel(logging.DEBUG)
    parser = DumpParser()
    try:
        count_values = 0
        for data in parser.parse(
            source=args.input,
            buffer_size=args.buffer_size,
            ignore_errors=not args.fail_on_error,
        ):
            json.dump(
                data,
                fp=args.output,
                ensure_ascii=False,
                cls=Base64Encoder,
            )
            args.output.write(os.linesep)
            args.output.flush()
            count_values += 1
        logger.info("Total values in %r: %d", args.input.name, count_values)
    except Exception as ex:
        logger.fatal(ex)
        return 1
    except KeyboardInterrupt:
        logger.warning("Aborted")


class Base64Encoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    def default(self, o: Any) -> str:
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return json.JSONEncoder.default(self, o)


if __name__ == "__main__":
    sys.exit(main())
