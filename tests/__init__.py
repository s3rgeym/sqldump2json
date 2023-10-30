import unittest

from sqldump2json import DumpParser


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DumpParser()

    def test(self) -> None:
        # Тут две запятые, после того как встречен какой-то неожиданный токен,
        # мы игнорируем все токены до следующего INSERT
        sql = """
INSERT/*lalalala*/INTO users(name,age)VALUES ('Petya', 31), ('Katya', 29),,
('Vasya', 42),
INSERT INTO users VALUES(DEFAULT, "Semyon", 52 / (4 - 2))
        """
        self.assertEqual(
            [*self.parser.parse(sql)],
            [
                {"table_name": "users", "values": {"age": 31, "name": "Petya"}},
                {"table_name": "users", "values": {"age": 29, "name": "Katya"}},
                {"table_name": "users", "values": [None, "Semyon", 26]},
            ],
        )
