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

![image](https://github.com/alacritty/alacritty/assets/12753171/c2dad1e7-7fd4-4c0d-a408-c85c4e0a1037)

Help:

```bash
sqldump2json -h
```

Values are converted to dict only if the `INSERT` contains a list of fields or the fields are declared in `CREATE TABLE`:

```sql
INSERT INTO users (id, name) VALUES (42, 'Vasyan');
```

Output:

```json
{"table_name": "users", "values": {"id": 42, "name": "Vasyan"}}
```

Convert to tsv:

```bash
$ sqldump2json -i tests/dump.sql | jq -r 'select(.table_name == "actor").values | @tsv'
1       PENELOPE        GUINESS 2006-02-15 04:34:33
2       NICK    WAHLBERG        2006-02-15 04:34:33
3       ED      CHASE   2006-02-15 04:34:33
...
```

Hex strings are converted to base64:

```bash
sqldump2json -i tests/dump.sql | tail -4 | head -1 | jq -r '.values[4]' | base64 -d > image.png
```

Supports basic arifmetic and boolean operations:

```bash
$ echo 'insert into test (result) values (-2 + 2 * 2);' | sqldump2json
{"table_name": "test", "values": {"result": 2}}
```

Cheat: use together with grep for table parsing:

```bash
grep 'CREATE TABLE `users`' /path/to/dump.sql | sqldump2json | jq
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
* Improve I/O performance. Тут хз что делать, потому как буфер не особо помогает. Токены так или иначе нужно читать посимвольно. Можно было разбирать целые строки, НО я смотрел дампы, у меня и отдельные строки занимают по 1.5 гигабайта, что, конечно, уже в раму влезет, но текущая реализация тем и хороша, что жрет не больше пары сотен мегабайт (зато грузит процессор). У меня есть дамп одного взломанного сайта, он весит 23 гига и в нем содержится 150 миллионов записей, если запустить его парсинг с выводом в консоль отладочной информации и буфером в 128K, он будет парситься 5 часов, что меня совсем не удовлетворяет, НО с ошибками парсер не падает с включенным флагом `--fail-on-error`. Тут, конечно, сложно сказать, что всему виной, потому как у меня Btrfs с высоким уровнем сжатия 9 и сам тот дамп реально на диске [ в сжатом виде ] занимет 3.2G, так что и мои тесты не совсем честные.
* Переименовать проект в parsedump?
