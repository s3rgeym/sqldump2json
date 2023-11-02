import doctest
import unittest
from pathlib import Path

import sqldump2json
from sqldump2json import DumpParser, logger

# logger.handlers.clear()

CUR_DIR = Path(__file__).parent
DUMP_FILE = CUR_DIR / "dump.sql"
CUCKOLD_PHOTO = CUR_DIR / "cuckold.png"


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DumpParser()

    def test_insert_values(self) -> None:
        sql = """
Любые невалидные токены вне `CREATE TABLE` и INSERT игнориуются

cReAte/**/taBLe users(
    user_id int NOT NULL,
    username varchar(31) NOT NULL,
    password varchar(31) NOT NULL,
    PRIMARY KEY(user_id));

insert into users values (1, 'tester', 'test123'),
    (2, 'dummyuser', '123456');

Тут проверяем многострочный комментарий, чтобы выражение внутри него не обработалось

/*/insert into users values ('xxx')/*/
INSERT INTO posts VALUES(123, "Hello World!", "Hello, world!");
        """
        self.assertEqual(
            # Кинет ошибку парсиннга, если внутри CREATE и INSERT встретит неожиданные токены
            [*self.parser.parse(sql, ignore_errors=False)],
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

    def test_insert_values_so39216133(self) -> None:
        sql = """
        DROP TABLE IF EXISTS `geo_tags`;
        /*!40101 SET @saved_cs_client     = @@character_set_client */;
        /*!40101 SET character_set_client = utf8 */;
        CREATE TABLE `geo_tags` (
          `gt_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
          `gt_page_id` int(10) unsigned NOT NULL,
          `gt_globe` varbinary(32) NOT NULL,
          `gt_primary` tinyint(1) NOT NULL,
          `gt_lat` decimal(11,8) DEFAULT NULL,
          `gt_lon` decimal(11,8) DEFAULT NULL,
          `gt_dim` int(11) DEFAULT NULL,
          `gt_type` varbinary(32) DEFAULT NULL,
          `gt_name` varbinary(255) DEFAULT NULL,
          `gt_country` binary(2) DEFAULT NULL,
          `gt_region` varbinary(3) DEFAULT NULL,
        PRIMARY KEY (`gt_id`),
        KEY `gt_page_primary` (`gt_page_id`,`gt_primary`),
        KEY `gt_page_id_id` (`gt_page_id`,`gt_id`)
        ) ENGINE=InnoDB AUTO_INCREMENT=4507036 DEFAULT CHARSET=binary ROW_FORMAT=DYNAMIC;

        INSERT INTO `geo_tags` VALUES (3,487781,'earth',1,59.00000000,10.00000000,1000,NULL,NULL,NULL,NULL);
        """
        self.assertEqual(
            next(self.parser.parse(sql)),
            {
                "table_name": "geo_tags",
                "values": {
                    "gt_country": None,
                    "gt_dim": 1000,
                    "gt_globe": "earth",
                    "gt_id": 3,
                    "gt_lat": 59.0,
                    "gt_lon": 10.0,
                    "gt_name": None,
                    "gt_page_id": 487781,
                    "gt_primary": 1,
                    "gt_region": None,
                    "gt_type": None,
                },
            },
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


# def load_tests(loader, tests, ignore):
#     tests.addTests(doctest.DocTestSuite(sqldump2json))
#     return tests
