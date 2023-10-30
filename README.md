# README

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqldump2json)]() [![PyPI - Version](https://img.shields.io/pypi/v/sqldump2json)]() [![Total Downloads](https://static.pepy.tech/badge/sqldump2json)]()

Parsing SQL dumps into JSON objects. A tool for administrators, data scientists and hackers. The dump is not read entirely into RAM, so this utility can be used to process files of any size. And it can even parse corrupted dumps.

Supported DBMS: MySQL, SQL Server, PotsgreSQL and some other (not all formats).

Installation:

```bash
# install pipx via your favorite package manager
pipx install sqldump2json
```

For crappy distros like Ubuntu you need to do more steps:

* Install pyenv or asdf-vm.
* Install latest python version and make it global via pyenv or asdf-vm.
* Install sqldump2json.
* OR use Docker.

Insert statements are converted to JSON objects on each new line:

```bash
$ sqldump2json -i tests/dump.sql
{"table_name": "actor", "values": [1, "PENELOPE", "GUINESS", "2006-02-15 04:34:33"]}
{"table_name": "actor", "values": [2, "NICK", "WAHLBERG", "2006-02-15 04:34:33"]}
{"table_name": "actor", "values": [3, "ED", "CHASE", "2006-02-15 04:34:33"]}
...
```

Use [jq](https://github.com/jqlang/jq) to process JSON (sort, filter and etc):

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

Supports basic arifmetic and boolean operations (i know it's useless):

```bash
$ echo 'insert into test (result) values (-2 + 2 * 2);' | sqldump2json
{"table_name": "test", "values": {"result": 2}}
```
