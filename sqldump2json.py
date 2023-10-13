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
import json
import logging
import os
import string
import sys
import typing
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Iterable, Iterator, Self, Sequence

__author__ = "Sergey M"


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
    ALTER = auto()
    AND = auto()
    ASSIGN = auto()
    BACKTICK_STRING = auto()
    BOOL = auto()
    COMMA = auto()
    COMMENT = auto()
    CONCAT = auto()
    CREATE = auto()
    DELETE = auto()
    DIV = auto()
    DOUBLE_QUOTED_STRING = auto()
    EMPTY = auto()
    EOF = auto()
    EQ = auto()
    EXISTS = auto()
    FLOAT = auto()
    FROM = auto()
    GT = auto()  # greater than
    GTE = auto()  # greater than or equal
    HEX_STRING = auto()
    IDENTIFIER = auto()
    IF = auto()
    INSERT = auto()
    INT = auto()
    INTO = auto()
    LPAREN = auto()
    LT = auto()  # less than
    LTE = auto()  # less than or equal
    MINUS = auto()
    MUL = auto()
    NE = auto()
    NOT = auto()
    NULL = auto()
    OR = auto()
    PERIOD = auto()
    PLUS = auto()
    QMARK = auto()
    RPAREN = auto()
    SELECT = auto()
    SEMICOLON = auto()
    SET = auto()
    STRING = auto()
    TABLE = auto()
    UPDATE = auto()
    USE = auto()
    VALUES = auto()
    WHERE = auto()
    WHITE_SPACE = auto()
    ...

    def __str__(self) -> str:
        return self.name


@dataclass
class Token:
    type: TokenType
    value: Any
    lineno: int = -1
    colno: int = -1


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


@dataclass
class Tokenizer:
    fp: typing.TextIO
    # ch: str | None = field(default=None, init=False)
    ASTERSISK_CHAR: ClassVar[str] = "*"
    BACKTICK_CHAR: ClassVar[str] = "`"
    DOUBLE_QUOTE_CHAR: ClassVar[str] = '"'
    ESCAPE_CHAR: ClassVar[str] = "\\"
    HEX_CHAR: ClassVar[str] = string.hexdigits
    HYPHEN_CHAR: ClassVar[str] = "-"
    IDENTIFIER_FIRST_CHAR: ClassVar[str] = "@" + string.ascii_letters + "_"
    IDENTIFIER_CHAR: ClassVar[str] = IDENTIFIER_FIRST_CHAR + string.digits
    NEWLINE_CHAR: ClassVar[str] = "\n"
    PERIOD_CHAR: ClassVar[str] = "."
    QUOTE_CHAR: ClassVar[str] = "'"
    SLASH_CHAR: ClassVar[str] = "/"
    TOKEN_EMPTY: ClassVar[Token] = Token(TokenType.EMPTY, "")
    SYMBOL_OPERATORS: ClassVar[dict[str, TokenType]] = {
        ":=": TokenType.ASSIGN,
        "!=": TokenType.NE,
        "<=": TokenType.LTE,
        "<>": TokenType.NE,
        ">=": TokenType.GTE,
        "||": TokenType.CONCAT,
        "-": TokenType.MINUS,  # substraction
        ",": TokenType.COMMA,
        ";": TokenType.SEMICOLON,
        ".": TokenType.PERIOD,
        "(": TokenType.LPAREN,
        ")": TokenType.RPAREN,
        "*": TokenType.MUL,
        "/": TokenType.DIV,
        "+": TokenType.PLUS,  # addition
        "<": TokenType.LT,
        "=": TokenType.EQ,
        ">": TokenType.GT,
    }

    def readch(self) -> str:
        c = self.fp.read(1)
        if c == self.NEWLINE_CHAR:
            self.lineno += 1
            self.colno = 1
        elif c:
            self.colno += 1
        return c

    # TODO: этот метод можно было бы использовать в других реализациях парсеров. В текущей есть ограничение, связанное с тем, что в пайпах нельзя сикать (можно использовать буфер, но мне лень переписывать).
    def peekch(self, n: int = 1) -> str:
        rv = self.fp.read(n)
        self.fp.seek(self.fp.tell() - len(rv))
        return rv

    def advance(self) -> None:
        self.prev_ch, self.ch, self.next_ch = (
            self.ch,
            self.next_ch,
            self.readch(),
        )

    def next_token(self) -> Token:
        # выглядит очень коряво, но мы не можем регулярками распарсить файл, тк тогда его нужно будет прочитать весь, а данная утилита нужна для парсинга гигабайтных-дампов
        self.advance()
        self.token_lineno = self.lineno
        self.token_colno = self.colno
        # TODO: вынести в отдельные методы
        # >>> '' in 'abc'
        # True
        if self.ch == "":
            return Token(TokenType.EOF, "")
        # пробельные символы
        if self.ch.isspace():
            val = self.ch
            while self.next_ch.isspace():
                self.advance()
                val += self.ch
            return Token(TokenType.WHITE_SPACE, val)
        # идентефикаторы, константы, операторы и ключевые слова
        if self.ch in self.IDENTIFIER_FIRST_CHAR:
            val = self.ch
            while self.next_ch in self.IDENTIFIER_CHAR:
                self.advance()
                val += self.ch
            upper = val.upper()
            match upper:
                case "NULL":
                    return Token(TokenType.NULL, None)
                case "TRUE" | "FALSE":
                    return Token(TokenType.BOOL, val == "TRUE")
                case (
                    "ALTER"
                    | "AND"
                    | "CREATE"
                    | "DELETE"
                    | "EXISTS"
                    | "FROM"
                    | "IF"
                    | "INSERT"
                    | "INTO"
                    | "NOT"
                    | "OR"
                    | "SELECT"
                    | "SET"
                    | "TABLE"
                    | "UPDATE"
                    | "USE"
                    | "VALUES"
                    | "WHERE"
                ):
                    return Token(TokenType[upper], val)
            return Token(TokenType.IDENTIFIER, val)
        # бинарные данные хранятся в Неведомой Ебанной Хуйне
        if self.ch == "0" and self.next_ch.lower() == "x":
            val = "0x"
            self.advance()
            while self.next_ch in self.HEX_CHAR:
                self.advance()
                val += self.ch.upper()
                # hex-строки могут разбиваться
                if self.next_ch == self.NEWLINE_CHAR:
                    self.advance()
            if len(val) == 2:
                raise Error(
                    f"invalid hex string at line {self.token_lineno} and column {self.token_colno}"
                )
            # TODO: конвертировать в bytes?
            return Token(TokenType.HEX_STRING, val)
        # числа
        if self.ch.isnumeric():
            val = self.ch
            while self.next_ch.isnumeric() or (
                self.next_ch == self.PERIOD_CHAR and not self.PERIOD_CHAR in val
            ):
                self.advance()
                val += self.ch
            return (
                Token(TokenType.FLOAT, float(val))
                if self.PERIOD_CHAR in val
                else Token(TokenType.INT, int(val))
            )
        # строки с разными типами кавычек
        for quote_char, token_type in [
            (self.QUOTE_CHAR, TokenType.STRING),
            (self.DOUBLE_QUOTE_CHAR, TokenType.DOUBLE_QUOTED_STRING),
            (self.BACKTICK_CHAR, TokenType.BACKTICK_STRING),
        ]:
            if self.ch == quote_char:
                # кавычки не нужно запоминать
                val = ""
                while True:
                    self.advance()
                    if not self.ch:
                        raise UnexpectedEnd(f"expected: {quote_char!r}")
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
                # return Token(token_type, ast.literal_eval(f'"{val}"'))
                return Token(token_type, val)
        # однострочный комментарий
        if self.ch == self.HYPHEN_CHAR == self.next_ch:
            val = self.ch * 2
            self.advance()
            while True:
                self.advance()
                if not self.ch:
                    break
                val += self.ch
                if self.ch == self.NEWLINE_CHAR:
                    break
            return Token(TokenType.COMMENT, val)
        # многострочный комментарий
        if self.ch == self.SLASH_CHAR and self.next_ch == self.ASTERSISK_CHAR:
            val = self.ch + self.next_ch
            self.advance()
            while True:
                self.advance()
                if not self.ch:
                    break
                val += self.ch
                # TODO: считает /*/ многострочным комментарием
                if (
                    self.prev_ch == self.ASTERSISK_CHAR
                    and self.ch == self.SLASH_CHAR
                ):
                    break
            return Token(TokenType.COMMENT, val)
        # символьные операторы
        for op, tt in self.SYMBOL_OPERATORS.items():
            assert 2 >= len(op) > 0
            if len(op) == 2:
                if self.ch + self.next_ch != op:
                    continue
                self.advance()
            elif op != self.ch:
                continue
            return Token(tt, op)
        raise UnexpectedChar(char=self.ch, lineno=self.lineno, colno=self.colno)

    def tokenize(self) -> Iterable[Token]:
        self.fp.seekable() and self.fp.seek(0)
        self.ch = None
        self.colno = 0
        self.lineno = 1
        self.next_ch = self.readch()
        while t := self.next_token():
            t.colno, t.lineno = self.token_colno, self.token_lineno
            yield t
            if t.type == TokenType.EOF:
                break

    # def __iter__(self) -> typing.Self:
    #     self.tokenize_it = iter(self.tokenize())
    #     return self

    # def __next__(self) -> Token:
    #     return next(self.tokenize_it)

    def __iter__(self) -> Iterator[Token]:
        yield from self.tokenize()


class ParseError(Error):
    pass


@dataclass
class Parser:
    tokenizer: Tokenizer

    def peek_token(self) -> Token:
        while True:
            # выглядит очень костыльно
            token = next(self.tokenizer_it, Tokenizer.TOKEN_EMPTY)
            logging.debug("peek %s", token.type)
            if token.type in (TokenType.WHITE_SPACE, TokenType.COMMENT):
                continue
            return token

    def advance_token(self) -> None:
        self.prev_token, self.cur_token, self.next_token = (
            self.cur_token,
            self.next_token,
            self.peek_token(),
        )

    def accept(self, *expected: TokenType) -> bool:
        if self.cur_token.type in expected:
            self.advance_token()
            return True
        return False

    def expect(self, *expected: TokenType) -> Self:
        if not self.accept(*expected):
            raise ParseError(
                f"unexpected token {self.cur_token.value!r} at line {self.cur_token.lineno} and column {self.cur_token.colno}; expected: {', '.join(map(str, expected))}"
            )
        # Когда нечего вернуть, то лучше всего возвращать self
        return self

    def quoted_identifier(self) -> str:
        return self.expect(
            TokenType.IDENTIFIER,
            TokenType.DOUBLE_QUOTED_STRING,
            TokenType.BACKTICK_STRING,
        ).prev_token.value

    def table_identifier(self) -> str:
        rv = self.quoted_identifier()
        # table_space.table_name
        if self.accept(TokenType.PERIOD):
            rv += self.prev_token.value + self.quoted_identifier()
        return rv

    def statement(self) -> None:
        # INSERT INTO tbl (col1, col2) VALUES ('a', 1);
        if self.accept(TokenType.INSERT):
            logging.debug("parse insert statement")
            self.expect(TokenType.INTO)
            table_name = self.table_identifier()
            # имена колонок опциональны
            column_names = []
            if self.accept(TokenType.LPAREN):
                while not self.accept(TokenType.RPAREN):
                    column_names.append(self.quoted_identifier())
                    if self.accept(TokenType.COMMA):
                        continue
            self.expect(TokenType.VALUES)
            while self.expect(TokenType.LPAREN):
                values = []
                while not self.accept(TokenType.RPAREN):
                    values.append(self.expression())
                    if self.accept(TokenType.COMMA):
                        continue
                json.dump(
                    {
                        # "statement": "insert",
                        "table_name": table_name,
                        "values": dict(zip(column_names, values))
                        if column_names
                        else values,
                    },
                    fp=sys.stdout,
                    ensure_ascii=False,
                )
                sys.stdout.write(os.linesep)
                sys.stdout.flush()
                if self.accept(TokenType.COMMA):
                    continue
                self.expect(TokenType.SEMICOLON)
                return
        while self.cur_token.type != TokenType.EOF:
            logging.debug("skip %s", self.cur_token.type)
            self.advance_token()
            if self.accept(TokenType.SEMICOLON):
                logging.debug("break ;")
                return

    def expression(self) -> Any:
        res = self.term()
        while self.accept(TokenType.PLUS, TokenType.MINUS):
            if self.prev_token.type == TokenType.PLUS:
                res += self.term()
            else:
                res -= self.term()
        return res

    def term(self) -> Any:
        res = self.unary()
        while self.accept(TokenType.MUL, TokenType.DIV):
            if self.prev_token.type == TokenType.MUL:
                res *= self.unary()
            else:
                res /= self.unary()
        return res

    def unary(self) -> Any:
        # +++1
        if self.accept(TokenType.PLUS):
            return +self.unary()
        if self.accept(TokenType.MINUS):
            return -self.unary()
        return self.primary()

    def primary(self) -> Any:
        if self.accept(TokenType.LPAREN):
            res = self.expression()
            self.expect(TokenType.RPAREN)
            return res
        return self.expect(
            TokenType.INT,
            TokenType.FLOAT,
            TokenType.BOOL,
            TokenType.NULL,
            TokenType.STRING,
            TokenType.HEX_STRING,
        ).prev_token.value

    def parse(self) -> None:
        self.tokenizer_it = iter(self.tokenizer)
        self.cur_token = Tokenizer.TOKEN_EMPTY
        self.next_token = self.peek_token()
        self.advance_token()
        while self.cur_token.type != TokenType.EOF:
            self.statement()
        logging.info("finished")


class NameSpace(argparse.Namespace):
    input: typing.TextIO
    debug: bool


def _parse_args(argv: Sequence[str] | None) -> NameSpace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", default="-", type=argparse.FileType())
    # parser.add_argument(
    #     "-o", "--output", default="-", type=argparse.FileType("w+")
    # )
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
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    parser = Parser(tokenizer=Tokenizer(args.input))
    try:
        parser.parse()
    except KeyboardInterrupt:
        print("\nGood bye!", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
