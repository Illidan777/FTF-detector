```
--------------------EDA--------------------------
```
- and also there is a case when available money  decreasing and in some moment available money equals credit limit again
- come up smth with account open date
- Find frequent transaction per similar account number (transaction date time)
```
--------------------Feature engineering--------------------------
```
- fill na with mode
- add feature that indicates about frequent address changing
- replace two cvv columns on matching marker feature
- sample data
- investigate about scaling ??
```
--------------------Model building--------------------------
```
- divide train test split
- investigate about hyperparameters tuning tools
- build models with benchmarks
- add pipeline to save models
- add controller to test transaction/group of transaction on fraud
```
--------------------There are some issues--------------------------
```
- ?? investigate how to process different types of anomalies (point, context, group)
- ?? Model types
- ?? Hybrid - cluster + classifier
- ?? Ensemble