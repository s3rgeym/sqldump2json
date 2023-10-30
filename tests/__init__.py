import unittest

from sqldump2json import DumpParser


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DumpParser()

    def test_inserts(self) -> None:
        sql = """
INSERT/*lalalala*/INTO users(name,age)VALUES ('Petya', 31),
('Vasay', 42), ('Semyon', 52 / (4 - 2))123
        """
        self.assertEqual(
            [*self.parser.parse(sql)],
            [
                {"table_name": "users", "values": {"age": 31, "name": "Petya"}},
                {"table_name": "users", "values": {"age": 42, "name": "Vasay"}},
                {
                    "table_name": "users",
                    "values": {"age": 26, "name": "Semyon"},
                },
            ],
        )
