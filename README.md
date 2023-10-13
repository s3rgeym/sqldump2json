# README

Parse SQL Dumps to JSON Objects. Tool for Administrators, Data Scientists and Hackers. Supported DBMS: MySQL, SQL Server, PotsgreSQL and other.

Installation:

```bash
# i recommend use pipx instead pip
pipx install sqldump2json
```

Insert statements are converted to JSON objects on each new line:

```bash
$ sqldump2json -i testdata/dump.sql
{"table_name": "actor", "values": [1, "PENELOPE", "GUINESS", "2006-02-15 04:34:33"]}
{"table_name": "actor", "values": [2, "NICK", "WAHLBERG", "2006-02-15 04:34:33"]}
{"table_name": "actor", "values": [3, "ED", "CHASE", "2006-02-15 04:34:33"]}
...
```

The file is not read entirely into RAM, so this utility can be used to process huge files.

Use [jq](https://github.com/jqlang/jq) to process JSON (sort, filter and etc):

```bash
$ ./sqldump2json -i testdata/dump.sql | jq -r 'select(.table_name == "actor").values | @tsv'
1       PENELOPE        GUINESS 2006-02-15 04:34:33
2       NICK    WAHLBERG        2006-02-15 04:34:33
3       ED      CHASE   2006-02-15 04:34:33
...
```
