#!/usr/bin/env python
# Тут реализована ЛИШЬ ЧАСТИЧНАЯ поддержка SQL — ТОЛЬКО INSERT выражений
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
import io
import json
import logging
import os
import string
import sys
import typing
from base64 import b64encode
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Self, Sequence, Type, TypedDict

__author__ = "Sergey M"


class Color(Enum):
    RESET = 0
    BLACK = 30
    RED = auto()
    GREEN = auto()
    YELLOW = auto()
    BLUE = auto()
    MAGENTA = auto()
    CYAN = auto()
    WHITE = auto()

    def __str__(self) -> str:
        return f"\033[{self.value}m"


class ColorHandler(logging.StreamHandler):
    COLOR_LEVELS = {
        logging.DEBUG: Color.BLUE,
        logging.INFO: Color.GREEN,
        logging.WARNING: Color.YELLOW,
        logging.ERROR: Color.RED,
        logging.CRITICAL: Color.MAGENTA,
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
logger.addHandler(ColorHandler())

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


class AutoName(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[Any],
    ) -> Any:
        return name


class TokenType(AutoName):
    T_ALTER = auto()
    T_AND = auto()
    T_ASSIGN = auto()
    T_BACKTICK_STRING = auto()
    T_BOOL = auto()
    T_COMMA = auto()
    T_COMMENT = auto()
    T_CONCAT = auto()
    T_CREATE = auto()
    # T_DELETE = auto()
    T_DIV = auto()
    T_DOUBLE_QUOTED_STRING = auto()
    T_EMPTY = auto()
    T_EOF = auto()
    T_EQ = auto()
    T_EXISTS = auto()
    T_FLOAT = auto()
    T_FROM = auto()
    T_GT = auto()  # greater than
    T_GTE = auto()  # greater than or equal
    T_HEX_STRING = auto()
    T_IDENTIFIER = auto()
    T_IF = auto()
    T_INSERT = auto()
    T_INT = auto()
    T_INTO = auto()
    T_LPAREN = auto()
    T_LT = auto()  # less than
    T_LTE = auto()  # less than or equal
    T_MINUS = auto()
    T_MUL = auto()
    T_NE = auto()
    T_NOT = auto()
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
    T_TABLE = auto()
    # T_UPDATE = auto()
    # T_USE = auto()
    T_VALUES = auto()
    T_WHERE = auto()
    T_WHITE_SPACE = auto()

    # Всякие операторы которым мне лень имена выдумывать
    T_DUMMY = auto()
    ...

    # Ошибки
    T_UNEXPECTED_CHAR = auto()
    T_INVALID_TOKEN = auto()

    def __str__(self) -> str:
        return self.value


class Token(typing.NamedTuple):
    type: TokenType
    value: Any
    lineno: int = -1
    colno: int = -1

    @property
    def repr_value(self) -> Any:
        return (
            cut(self.value, 31) if isinstance(self.value, str) else self.value
        )

    def __repr__(self) -> str:
        return f"token {self.repr_value!r} ({self.type}) at line {self.lineno} and column {self.colno}"


def cut(s: str, n: int, dots: str = "...") -> str:
    """
    >>> cut("Привет, мир!", 10)
    'Привет,...'
    """
    return s[: n - len(dots)] + ["", dots][len(s) > n]


class Error(Exception):
    pass


@dataclass
class UnexpectedChar(Error):
    char: str
    lineno: int
    colno: int

    def __str__(self) -> str:
        return f"unexcpected char {self.char!r} at line {self.lineno} and column {self.colno}"


class UnexpectedEnd(Error):
    pass


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
    TOKEN_EMPTY = Token(TokenType.T_EMPTY, "")
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

    def __init__(self, input: io.TextIOBase | str) -> None:
        self.input = io.StringIO(input) if isinstance(input, str) else input

    # Медленная функция
    def readch(self) -> str:
        c = self.input.read(1)
        if c == self.NEWLINE_CHAR:
            self.lineno += 1
            self.colno = 0
        elif c:
            self.colno += 1
        return c

    # TODO: этот метод можно было бы использовать в других реализациях парсеров. В текущей есть ограничение, связанное с тем, что в пайпах нельзя сикать (можно использовать буфер, но мне лень переписывать).
    def peekch(self, n: int = 1) -> str:
        rv = self.input.read(n)
        self.input.seek(self.input.tell() - len(rv))
        return rv

    def advance(self) -> None:
        self.ch, self.peek_ch = (
            self.peek_ch,
            self.readch(),
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
            return self.token(TokenType.T_EOF, "")
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
                    return self.token(TokenType.T_BOOL, val == "TRUE")
                case "AND" | "INSERT" | "INTO" | "NOT" | "OR" | "VALUES":
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
        if (
            self.ch == self.MINUS_CHAR and self.peek_ch.isnumeric()
        ) or self.ch.isnumeric():
            val = self.ch
            if self.ch == self.MINUS_CHAR:
                self.advance()
                val += self.ch
            while self.peek_ch.isnumeric() or (
                self.peek_ch == self.PERIOD_CHAR and not self.PERIOD_CHAR in val
            ):
                self.advance()
                val += self.ch
            return (
                self.token(TokenType.T_FLOAT, float(val))
                if self.PERIOD_CHAR in val
                else self.token(TokenType.T_INT, int(val))
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
                    if self.ch == self.ESCAPE_CHAR:
                        self.advance()
                        match self.ch:
                            case "0":
                                val += "\0"
                            case "n":
                                val += "\n"
                            case "r":
                                val += "\r"
                            case "t":
                                val += "\t"
                            case _:
                                val += self.ch
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
        getattr(self.input, "seekable", lambda: 0)() and self.input.seek(0)
        self.ch = None
        self.colno = 0
        self.lineno = 1
        self.peek_ch = self.readch()
        while t := self.next_token():
            # if t.type in (TokenType.T_WHITE_SPACE, TokenType.T_COMMENT):
            #     continue
            logger.debug(t)
            yield t
            if t.type == TokenType.T_EOF:
                break

    __iter__ = tokenize


class ParseError(Error):
    pass


class InsertValues(TypedDict):
    table_name: str
    values: dict[str, Any] | list[Any]


@dataclass
class DumpParser:
    tokenizer_class: Type = SQLTokenizer

    def advance_token(self) -> None:
        self.cur_token, self.next_token = (
            self.next_token,
            next(self.tokenizer_it, None),
        )
        if self.cur_token is None and self.next_token is None:
            raise ParseError("unexpected end")

    def peek_token(self, *expected: TokenType) -> bool:
        """Проверить тип следующего токена и сделать его текущим"""
        if self.next_token.type in expected:
            self.advance_token()
            return True
        return False

    def expect_token(self, *expected: TokenType) -> Self:
        if not self.peek_token(*expected):
            raise ParseError(f"unexpected: {self.next_token}")
        # Когда нечего вернуть, то лучше всего возвращать self
        return self

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
        logger.debug("parse insert statement")
        self.expect_token(TokenType.T_INTO)
        table_name = self.table_identifier()
        # имена колонок опциональны
        column_names = []
        if self.peek_token(TokenType.T_LPAREN):
            while True:
                column_names.append(self.quoted_identifier())
                if self.peek_token(TokenType.T_RPAREN):
                    break
                # rparen просто для красоты
                self.expect_token(TokenType.T_COMMA, TokenType.T_RPAREN)
        self.expect_token(TokenType.T_VALUES)
        while self.expect_token(TokenType.T_LPAREN):
            values = []
            while True:
                values.append(self.expr())
                if self.peek_token(TokenType.T_RPAREN):
                    break
                self.expect_token(TokenType.T_COMMA, TokenType.T_RPAREN)
            yield {
                "table_name": table_name,
                "values": dict(zip(column_names, values))
                if column_names
                else values.copy(),
            }
            if self.peek_token(TokenType.T_COMMA):
                continue
            self.expect_token(TokenType.T_SEMICOLON)
            return

    # Проритет операторов описан здесь:
    # https://learn.microsoft.com/ru-ru/sql/t-sql/language-elements/operator-precedence-transact-sql?view=sql-server-ver16
    # Чем ниже приоритет тем первее вызывается
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
        # values (default, ...)
        if self.peek_token(TokenType.T_IDENTIFIER):
            return None
        return self.expect_token(
            TokenType.T_INT,
            TokenType.T_FLOAT,
            TokenType.T_BOOL,
            TokenType.T_NULL,
            TokenType.T_STRING,
            TokenType.T_DOUBLE_QUOTED_STRING,  # В MySQL можно
            TokenType.T_HEX_STRING,
            TokenType.T_IDENTIFIER,
        ).cur_token.value

    def parse(
        self,
        source: typing.TextIO | str,
        ignore_errors: bool = True,
    ) -> Iterable[InsertValues]:
        self.tokenizer = self.tokenizer_class(input=source)
        self.tokenizer_it = iter(self.tokenizer)
        self.cur_token = self.next_token = None
        self.advance_token()
        while not self.peek_token(TokenType.T_EOF):
            if self.peek_token(TokenType.T_INSERT):
                try:
                    yield from self.parse_insert()
                # CREATE TRIGGER customer_create_date BEFORE INSERT ON customer
                # 	FOR EACH ROW SET NEW.create_date = NOW();
                except ParseError as ex:
                    if not ignore_errors:
                        raise ex
                    logger.warning(ex)
                continue
            self.advance_token()
        logger.info("finished")


class NameSpace(argparse.Namespace):
    input: typing.TextIO
    output: typing.TextIO
    debug: bool


def _parse_args(argv: Sequence[str] | None) -> NameSpace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default="-", type=argparse.FileType())
    parser.add_argument(
        "-o", "--output", default="-", type=argparse.FileType("w+")
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="show debug messages",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args(argv, namespace=NameSpace())
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logger.setLevel(level=(logging.WARNING, logging.DEBUG)[args.debug])
    parser = DumpParser()
    try:
        for v in parser.parse(args.input):
            json.dump(
                v,
                fp=args.output,
                ensure_ascii=False,
                cls=Base64Encoder,
            )
            args.output.write(os.linesep)
            args.output.flush()
    except KeyboardInterrupt:
        print("\nbye!", file=sys.stderr)


class Base64Encoder(json.JSONEncoder):
    # pylint: disable=method-hidden
    def default(self, o: Any) -> str:
        if isinstance(o, bytes):
            return b64encode(o).decode()
        return json.JSONEncoder.default(self, o)


if __name__ == "__main__":
    sys.exit(main())
