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
import dataclasses
import functools
import io
import itertools
import logging
import os
import re
import string
import sys
import typing
from base64 import b64encode
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, IntFlag, auto
from typing import Any, Iterable, Self, Sequence, Type, TypedDict

__author__ = "Sergey M"


def default_json(o: Any) -> str:
    if isinstance(o, bytes):
        return b64encode(o).decode()
    raise TypeError


try:
    # Не оправдал ожиданий
    import orjson

    def dump_json(obj: Any, fp: io.TextIOBase) -> None:
        data = orjson.dumps(obj, default=default_json)
        # fp.write(data.decode())
        fp.buffer.write(data)

except ImportError:
    import json

    dump_json = functools.partial(
        json.dump, ensure_ascii=True, default=default_json
    )


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
        logging.DEBUG: Color.BRIGHT_CYAN,
        logging.INFO: Color.BRIGHT_GREEN,
        logging.WARNING: Color.BRIGHT_YELLOW,
        logging.ERROR: Color.BRIGHT_RED,
        logging.CRITICAL: Color.BRIGHT_MAGENTA,
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
# class AutoName(str, Enum):
#     @staticmethod
#     def _generate_next_value_(
#         name: str,
#         start: int,
#         count: int,
#         last_values: list[Any],
#     ) -> Any:
#         return name


class TokenType(IntFlag):
    # Вместо кучи разных T_INSERT и T_UPDATE можно использовать T_KEYWORD или T_OPERATOR
    # Исп в INSERT
    T_AND = auto()
    T_ASSIGN = auto()
    T_BACTICK_QUOTED = auto()
    T_BOOL = auto()
    T_COMMA = auto()
    T_CONCAT = auto()
    T_DIV = auto()
    T_DOUBLE_QUOTED = auto()
    T_EOF = auto()
    T_EQ = auto()
    # T_FLOAT = auto()
    T_FROM = auto()
    T_GT = auto()  # greater than
    T_GTE = auto()  # greater than or equal
    T_HEX = auto()
    T_ID = auto()
    T_INSERT = auto()
    # T_INT = auto()
    T_INTO = auto()
    T_LPAREN = auto()
    T_LT = auto()  # less than
    T_LTE = auto()  # less than or equal
    T_MINUS = auto()
    T_MUL = auto()
    T_NE = auto()
    T_NULL = auto()
    T_NUMBER = auto()
    T_OR = auto()
    T_PERIOD = auto()
    T_PLUS = auto()
    T_QMARK = auto()
    T_QUOTED = auto()
    T_RPAREN = auto()
    T_SEMI = auto()
    T_VALUES = auto()
    T_STRING = T_QUOTED | T_DOUBLE_QUOTED | T_HEX
    T_SCALAR = T_NUMBER | T_BOOL | T_NULL | T_STRING
    # https://docs.jboss.org/author/display/TEIID/BNF%20for%20SQL%20Grammar.html#18646304_BNFforSQLGrammar-tokenQUOTEDID
    # `mytable` используется в MySQL/Persona/MariaDB
    T_QUOTED_ID = T_BACTICK_QUOTED | T_DOUBLE_QUOTED | T_ID

    # Исп в CREATE
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

    # Смена базы
    T_USE = auto()
    T_SET = auto()
    T_TO = auto()

    # Ошибки
    T_UNEXPECTED_CHAR = auto()
    T_INVALID_TOKEN = auto()

    # Костыль
    T_EMPTY = auto()

    def __str__(self) -> str:
        return self.name


class Token(typing.NamedTuple):
    type: TokenType
    value: Any = ""
    lineno: int = -1
    colno: int = -1

    @property
    def repr_value(self) -> Any:
        return (
            "**BINARY DATA**"
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
    HEX_CHARS = string.hexdigits
    MINUS_CHAR = "-"
    ID_FIRST_CHARS = "#@" + string.ascii_letters + "_"
    ID_CHARS = ID_FIRST_CHARS + string.digits
    NEWLINE_CHAR = "\n"
    PERIOD_CHAR = "."
    QUOTE_CHAR = "'"
    SLASH_CHAR = "/"
    SYMBOL_OPERATORS: dict[str, TokenType] = {
        "-": TokenType.T_MINUS,  # substraction
        "-=": TokenType.T_DUMMY,
        ",": TokenType.T_COMMA,
        ";": TokenType.T_SEMI,
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
        if self.ch in self.ID_FIRST_CHARS:
            val = self.ch
            while self.peek_ch in self.ID_CHARS:
                self.advance()
                val += self.ch
            upper = val.upper()
            match upper:
                case "NULL":
                    return self.token(TokenType.T_NULL, None)
                case "TRUE" | "FALSE":
                    return self.token(TokenType.T_BOOL, val == "TRUE")
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
                    | "SET"
                    | "SPATIAL"
                    | "TABLE"
                    | "TO"
                    | "UNIQUE"
                    | "USE"
                    | "VALUES"
                ):
                    return self.token(TokenType[f"T_{upper}"], val)
            return self.token(TokenType.T_ID, val)
        # бинарные данные хранятся в Неведомой Ебанной Хуйне
        if self.ch == "0" and self.peek_ch.lower() == "x":
            val = ""
            self.advance()
            while self.peek_ch in self.HEX_CHARS:
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
            return self.token(TokenType.T_HEX, binascii.unhexlify(val))
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
            val = float(val) if self.PERIOD_CHAR in val else int(val)
            return self.token(TokenType.T_NUMBER, val)
        # строки с разными типами кавычек
        for quote_char, token_type in [
            (self.QUOTE_CHAR, TokenType.T_QUOTED),
            (self.DOUBLE_QUOTE_CHAR, TokenType.T_DOUBLE_QUOTED),
            (self.BACKTICK_CHAR, TokenType.T_BACTICK_QUOTED),
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
    table: str
    schema: typing.NotRequired[str]
    values: dict[str, Any] | list[Any]


MISSING = object()


@dataclass
class DumpParser:
    _: dataclasses.KW_ONLY
    ignore_errors: bool = True
    tokenizer_class: Type = SQLTokenizer

    def advance_token(self) -> None:
        self.token, self.next_token = (
            self.next_token,
            # Этот пустой токен тупо костыль, но альтернативу ему я не придумал
            next(self.tokenizer_it, Token(TokenType.T_EMPTY)),
        )

    def peek_token(self, tt: TokenType, val: Any = MISSING) -> bool:
        """Проверить тип и значение следующего токена и если они совпадают, сделать его текущим"""
        if (self.next_token.type & tt) == self.next_token.type and (
            val is MISSING or val == self.next_token.value
        ):
            logger.debug("peek %s", self.next_token)
            self.advance_token()
            return True
        return False

    def expect_token(self, tt: TokenType, val: Any = MISSING) -> Self:
        if not self.peek_token(tt, val):
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
            x = self.AND()
            rv = rv or x
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
        while self.peek_token(TokenType.T_MUL | TokenType.T_DIV):
            if self.token.type == TokenType.T_MUL:
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
        if self.peek_token(TokenType.T_ID):
            return None
        return self.expect_token(TokenType.T_SCALAR).token.value

    def quoted_id(self) -> str:
        return self.expect_token(TokenType.T_QUOTED_ID).token.value

    # Это неправильно
    def table_name(self) -> tuple[str, str]:
        ident = self.quoted_id()
        if self.peek_token(TokenType.T_PERIOD):
            return self.quoted_id(), ident
        return ident, None

    def parse_insert(self) -> Iterable[InsertValues]:
        # INSERT INTO tbl (col1, col2) VALUES ('a', 1);
        logger.debug("parse insert values")
        table_name, table_schema = self.table_name()
        logger.debug(f"{table_name=}; {table_schema=}")
        # имена колонок опциональны
        column_names = []
        if self.peek_token(TokenType.T_LPAREN):
            while True:
                column_names.append(self.quoted_id())
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
            table_schema = table_schema or self.table_schema
            column_names = self.table_fields[table_schema].get(
                table_name,
                column_names,
            )
            if column_names:
                if len(column_names) != len(values):
                    raise ParseError(
                        f"missmatch number of column names and number of values for table {table_name!r}"
                    )
                values = dict(zip(column_names, values))
            yield {
                "table": table_name,
                "schema": table_schema,
                "values": values.copy(),
            }
            # counter += 1
            if self.peek_token(TokenType.T_COMMA):
                continue
            self.expect_token(TokenType.T_SEMI)
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
        table_name, schema = self.table_name()
        logger.debug(f"{table_name=}")
        self.expect_token(TokenType.T_LPAREN)
        # помним, что в дампе может быть несколько таблиц с одинаковым именем
        column_names = self.table_fields[schema or self.table_schema][
            table_name
        ] = []
        while True:
            # Я не все скорее всего перечислил ключевые слова для объявления индексов отдельно
            if not self.peek_token(
                TokenType.T_CHECK
                | TokenType.T_CONSTRAINT
                | TokenType.T_FOREIGN
                | TokenType.T_FULLTEXT
                | TokenType.T_INDEX
                | TokenType.T_KEY
                | TokenType.T_PRIMARY
                | TokenType.T_SPATIAL
                | TokenType.T_UNIQUE
            ):
                column_name = self.quoted_id()
                column_names.append(column_name)
            # Проверяем формальную правильность синтаксиса
            while not self.peek_token(TokenType.T_COMMA):
                if self.peek_token(TokenType.T_RPAREN):
                    logger.debug(
                        "%r.%r: %r",
                        schema,
                        table_name,
                        column_names,
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
        buffer_size: int = 8_192,
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
        self.table_fields = defaultdict(dict)
        self.table_schema = None
        while not self.peek_token(TokenType.T_EOF | TokenType.T_EMPTY):
            try:
                # MySQL, SQL Server
                # use database
                if self.peek_token(TokenType.T_USE):
                    self.table_schema = self.quoted_id()
                    self.expect_token(TokenType.T_SEMI)
                # PostgreSQL
                # set search_path='public';
                elif self.peek_token(TokenType.T_SET):
                    # TODO: посмотреть какой синтаксис правильный
                    if self.peek_token(
                        TokenType.T_QUOTED_ID, "search_path"
                    ) and self.peek_token(TokenType.T_EQ | TokenType.T_TO):
                        self.table_schema = self.expect_token(
                            TokenType.T_STRING
                        ).token.value
                        self.expect_token(TokenType.T_SEMI)
                elif self.peek_token(TokenType.T_CREATE):
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

    __call__ = parse


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
    s: str,
    *,
    units: typing.Sequence[str] = "KMG",
    base: int = 1024,
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


def skip_none(o: dict) -> dict:
    return {k: v for k, v in o.items() if v is not None}


def main(argv: Sequence[str] | None = None) -> int | None:
    args = _parse_args(argv)
    # logger тормозная вещь так как блокирует основной поток
    logger.addHandler(ColorHandler())
    if args.debug:
        logger.setLevel(logging.DEBUG)
    try:
        parse = DumpParser()
        count = 0
        for count, item in enumerate(map(
            skip_none,
            parse(
                source=args.input,
                buffer_size=args.buffer_size,
                ignore_errors=not args.fail_on_error,
            ),
        ), 1):
            dump_json(item, args.output)
            args.output.write(os.linesep)
            args.output.flush()
        logger.info("Total values in %r: %d", args.input.name, count)
    except Exception as ex:
        logger.exception(ex)
        return 1
    except KeyboardInterrupt:
        logger.warning("Aborted")


if __name__ == "__main__":
    sys.exit(main())
