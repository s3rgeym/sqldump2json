import unittest
from pathlib import Path

from sqldump2json import DumpParser

CUR_DIR = Path(__file__).parent
DUMP_FILE = CUR_DIR / "dump.sql"
CUCKOLD_PHOTO = CUR_DIR / "cuckold.png"


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DumpParser()

    def test_corrupted_data(self) -> None:
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

    def test_hex_string2bytes(self) -> None:
        with DUMP_FILE.open() as fp:
            for res in self.parser.parse(fp):
                if res["table_name"] != "staff":
                    continue
                first_name, last_name, _, img_data = res["values"][1:5]
                # Там фотка этого васяна
                if first_name == "Mike" and last_name == "Hillyer":
                    self.assertEqual(b"\x89PNG", img_data[:4])
                    # Убедимся что на фото грязный американский, заплывший жиром куколд бургер-Джо с дилдо в его радужной жопе
                    self.assertEqual(
                        img_data,
                        CUCKOLD_PHOTO.read_bytes(),
                    )
                    return
        self.fail()
