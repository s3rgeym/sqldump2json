# README

Parse SQL Dumps to JSON Objects.

Supported DBMS: MySQL, SQL Server, PotsgreSQL.

The file is not read entirely into RAM, so this utility can be used to process huge files.

Use [jq](https://github.com/jqlang/jq) to process JSON. Can be used for processing huge dumps:

```bash
sqlsump2json -i dump.sql | jq
```
