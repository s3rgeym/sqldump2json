# sqldump2json

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqldump2json)]() [![PyPI - Version](https://img.shields.io/pypi/v/sqldump2json)]() [![Total Downloads](https://static.pepy.tech/badge/sqldump2json)]()

Parse SQL dumps into JSON objects.

A tool for administrators, data scientists and hackers. With this tool you no longer need to import dumps into Databases. You can extract INSERT data as JSON and analyze them with [jq](https://github.com/jqlang/jq) or insert into Mongo/Elastic/etc. The dump is not read entirely into RAM, so this utility can be used to process files of any size. And it can even parse corrupted dumps.

Supported DBMS: MySQL, SQL Server, PotsgreSQL and some other (not all formats).

Installation for normal Arch-based Linux ditros:

```bash
# install pipx
yay -S python-pipx

# install package from pypi
pipx install sqldump2json

# install lastet version from github
pipx install git+https://github.com/s3rgeym/sqldump2json.git
```

For other shit like Ubuntu you need to do more steps:

* Install pyenv or asdf-vm.
* Install latest python version and make it global via pyenv or asdf-vm.
* Install sqldump2json.
* Or use Docker.

## CLI

Usage:

```bash
sqldump2json [ -h ] [ -i INPUT ] [ -o OUTPUT ] [ ... ]
```

Output format is JSONL:

```bash
echo "INSERT INTO data VALUES (1, 'foo'), (2, 'bar'), (3, 'baz');" | sqldump2json
{"table_name": "data", "values": [1, "foo"]}
{"table_name": "data", "values": [2, "bar"]}
{"table_name": "data", "values": [3, "baz"]}
```

Values are converted to dict only if the `INSERT INTO` contains a list of fields or the fields are declared in `CREATE TABLE`:

```bash
$ sqldump2json <<< "INSERT INTO data VALUES (NULL, 3.14159265, FALSE, 'Привет', 0xDEADBEEF);" | jq
{
  "table_name": "data",
  "values": [
    null,
    3.14159265,
    false,
    "Привет",
    "3q2+7w=="
  ]
}

$ sqldump2json <<< 'INSERT INTO `page` (title, contents) VALUES ("Title", "Text goes here");' | jq
{
  "table_name": "page",
  "values": {
    "title": "Title",
    "contents": "Text goes here"
  }
}
```

Using together with grep:

```bash
grep 'INSERT INTO `users`' /path/to/dump.sql | sqldump2json | jq -r '.values | [.username, .password] | @tsv' > output.csv
```

Also supports basic arifmetic and boolean operations:

```bash
$ echo 'insert into test (result) values (-2 + 2 * 2);' | sqldump2json
{"table_name": "test", "values": {"result": 2}}
```

## Scripting

If you were looking for a way how to import data from SQL to NoSQL databases and etc:

```python
#!/usr/bin/env python
from sqldump2json import DumpParser
...
if __name__ == '__main__':
    parser = DumpParser()
    for item in parser.parse("/path/to/dump.sql"):
        do_something(item)
```

## Development

Run tests:

```bash
poetry run python -m unittest
```

## TODO LIST

* Add support [mysql strings with charset](https://dev.mysql.com/doc/refman/8.0/en/charset-introducer.html) (eg, `_binary '\x00...'`). + `X'...'`
* Строки должны конкатенироваться, если идут подряд.
* Improve I/O performance. Тут хз что делать, потому как буфер не особо помогает. Токены так или иначе нужно читать посимвольно. Можно было разбирать целые строки, НО я смотрел дампы, у меня и отдельные строки занимают по 1.5 гигабайта, что, конечно, уже в раму влезет, но текущая реализация тем и хороша, что жрет не больше пары сотен мегабайт (зато грузит процессор). У меня есть дамп одного взломанного сайта, он весит 23 гига и в нем содержится почти 160 миллионов записей, если запустить его парсинг без вывода в консоль отладочной информации (!это очень важно, тк вывод в консоль сильно тормозит) с дополнительными флагами `-b128k -o /dev/null`, он будет парситься 5 часов, что меня не очень устраивает, поэтому советую использовать совместно с grep. Тут, конечно, негативно влияет и использование Btrfs со сжатием (сжатый дамп на диске занимает всего 3.6G). От игры с размером буфера я значительного прироста скорости парсинга не заметил (возможно, эта сомнительная фича будет удалена). Ускорения получиться добиться только при переписании проекта на что-то другое. Я пробовал через [codon](https://github.com/exaloop/codon) его запускать, то тот много чего не поддерживает, например, `dataclasses`.
* Переименовать проект в parsedump?
