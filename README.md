# README

Parse SQL Dumps to JSON Objects.

Insert statements are converted to JSON objects on each new line:

```bash
$ sqldump2json -i dump.sql                                                  
{"table_name": "actor", "items": [1, "PENELOPE", "GUINESS", "2006-02-15 04:34:33"]}
{"table_name": "actor", "items": [2, "NICK", "WAHLBERG", "2006-02-15 04:34:33"]}
{"table_name": "actor", "items": [3, "ED", "CHASE", "2006-02-15 04:34:33"]}
{"table_name": "actor", "items": [4, "JENNIFER", "DAVIS", "2006-02-15 04:34:33"]}
{"table_name": "actor", "items": [5, "JOHNNY", "LOLLOBRIGIDA", "2006-02-15 04:34:33"]}
```

Supported DBMS: MySQL, SQL Server, PotsgreSQL.

The file is not read entirely into RAM, so this utility can be used to process huge files.

Use [jq](https://github.com/jqlang/jq) to process JSON:

```bash
$ sqlsump2json -i dump.sql | jq
```
