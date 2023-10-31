import unittest
from pathlib import Path

from sqldump2json import DumpParser

CUR_DIR = Path(__file__).parent
DUMP_FILE = CUR_DIR / "dump.sql"
CUCKOLD_PHOTO = CUR_DIR / "cuckold.png"


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DumpParser()

    def test_insert_values(self) -> None:
        sql = """
cReAte/**/taBLe users(
    user_id int NOT NULL,
    username varchar(31) NOT NULL,
    password varchar(31) NOT NULL,
    PRIMARY KEY(user_id));

ошибки парсинга должны игнорироваться

insert into users values (1, 'tester', 'test123'),
    (2, 'dummyuser', '123456')

INSERT INTO posts VALUES(123, "Hello World!", "Hello, world!");
        """
        self.assertEqual(
            [*self.parser.parse(sql)],
            [
                {
                    "table_name": "users",
                    "values": {
                        "user_id": 1,
                        "username": "tester",
                        "password": "test123",
                    },
                },
                {
                    "table_name": "users",
                    "values": {
                        "user_id": 2,
                        "username": "dummyuser",
                        "password": "123456",
                    },
                },
                {
                    "table_name": "posts",
                    "values": [123, "Hello World!", "Hello, world!"],
                },
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
                    # Убедимся что на фото грязный американский, заплывший жиром, куколд Бургер Джо с дилдо в его радужной жопе
                    self.assertEqual(
                        img_data,
                        CUCKOLD_PHOTO.read_bytes(),
                    )
                    return
        self.fail()
