# sqldump2json

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqldump2json)]() [![PyPI - Version](https://img.shields.io/pypi/v/sqldump2json)]() [![Total Downloads](https://static.pepy.tech/badge/sqldump2json)]()

Parse SQL dumps into JSON objects.

A tool for administrators, data scientists and hackers. With this tool you no longer need to import dumps into Databases. You can extract INSERT data as JSON and analyze them with [jq](https://github.com/jqlang/jq). The dump is not read entirely into RAM, so this utility can be used to process files of any size. And it can even parse corrupted dumps.

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
* OR use Docker.

Insert statements are converted to JSON objects on each new line (JSONL):

```bash
$ sqldump2json -i tests/dump.sql
{"table_name": "actor", "values": [1, "PENELOPE", "GUINESS", "2006-02-15 04:34:33"]}
{"table_name": "actor", "values": [2, "NICK", "WAHLBERG", "2006-02-15 04:34:33"]}
{"table_name": "actor", "values": [3, "ED", "CHASE", "2006-02-15 04:34:33"]}
...
```

Filter and convert to TSV:

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

Values are converted to dict only if the `INSERT` contains a list of fields or the fields are declared in `CREATE TABLE`:

```sql
INSERT INTO users (id, name) VALUES (42, 'Vasyan');
```

Output:

```json
{"table_name": "users", "values": {"id": 42, "name": "Vasyan"}}
```

Supports basic arifmetic and boolean operations:

```bash
$ echo 'insert into test (result) values (-2 + 2 * 2);' | sqldump2json
{"table_name": "test", "values": {"result": 2}}
```
