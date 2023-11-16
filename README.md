# sqldump2json

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqldump2json)]() [![PyPI - Version](https://img.shields.io/pypi/v/sqldump2json)]() [![Total Downloads](https://static.pepy.tech/badge/sqldump2json)]()

Converts SQL dump to a JSON stream.

A tool for administrators, data scientists and hackers. With this tool you no longer need to import dumps into Databases. You can extract INSERT data as JSON and analyze them with [jq](https://github.com/jqlang/jq) or insert into MongoDB/Elastic/etc. The dump is not read entirely into RAM, so this utility can be used to process files of any size. And it can even parse corrupted dumps. No dependencies!

Supported DBMS: MySQL, SQL Server, PotsgreSQL and some other (not all formats).

RESTRICTIONS:

- Syntax is checked only for `INSERT INTO` and `CREATE TABLE`.
- The common SQL syntax is used which does not fully correspond to either MySQL or Postgres.
- Function calls and subquieries in INSERT satetements are not supported.

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

- Install pyenv or asdf-vm.
- Install latest python version and make it global via pyenv or asdf-vm.
- Install sqldump2json.
- Or use Docker.

## CLI

Usage:

```bash
sqldump2json [ -h ] [ -i INPUT ] [ -o OUTPUT ] [ ... ]
```

Output format is JSONL:

```bash
echo "INSERT INTO db.data VALUES (1, 'foo'), (2, 'bar'), (3, 'baz');" | sqldump2json
{"table": "data", "schema": "db", "values": [1, "foo"]}
{"table": "data", "schema": "db", "values": [2, "bar"]}
{"table": "data", "schema": "db", "values": [3, "baz"]}
```

Values are converted to dict only if the `INSERT INTO` contains a list of fields or the fields are declared in `CREATE TABLE`:

```bash
$ sqldump2json <<< "INSERT INTO data VALUES (NULL, 3.14159265, FALSE, 'Привет', 0xDEADBEEF);" | jq
{
  "table": "data",
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
  "table": "page",
  "values": {
    "title": "Title",
    "contents": "Text goes here"
  }
}
```

Using together with grep:

```bash
grep 'INSERT INTO `users`' /path/to/dump.sql | sqldump2json | jq -r '.values | [.username, .email, .password] | @tsv' > output.csv
```

## Scripting

If you were looking for a way how to import data from SQL to NoSQL databases and etc:

```python
#!/usr/bin/env python
from sqldump2json import DumpParser
...
if __name__ == '__main__':
    parse = DumpParser()
    for val in parse("/path/to/dump.sql"):
        do_something(val)
```

## Development

Run tests:

```bash
poetry run python -m uniзщttest
```

## TODO LIST

- Add support [mysql strings with charset](https://dev.mysql.com/doc/refman/8.0/en/charset-introducer.html) (eg, `_binary '\x00...'`). + `X'...'`
- Строки должны конкатенироваться, если идут подряд.

## Notes

После создания этого пакета я случайно узнал про существование [sqldump-to](https://github.com/arjunmehta/sqldump-to). Тот проект заброшен, и та утилита НЕ МОЖЕТ ПАРСИТЬ ДАМПЫ ПО 100500 ГИГАБАЙТ.

Я пробовал ускорить парсинг с помощью orjson (реализован на говнорасте и отвечает за парсинг JSON), но вопреки заявленному ускорению в 10 раз, получил замедление при парсинге 23-гигового дампа, содержащего 160 миллинов вставок, с 5 часов до 7.
