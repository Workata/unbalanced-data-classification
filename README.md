# unbalanced-data-classification
## Development

### Setup project

<!-- * Create/copy config file:
```sh
cp config_example.json config.json
``` -->

<!-- * Change config settings if needed -->

* Create venv
```sh
python3 -m venv ./venv
```

* Activate venv
```sh
. ./venv/bin/activate
```

* Install libs
```sh
pip install -r ./requirements/dev.txt
```

* Download listed datasets and place them inside `datasets` folder

* Run program
```py
python3 ./src/main.py
```

### Test code

* Linter (flake8)
```sh
flake8 ./src/
```

<!-- * Unit tests (pytest)
```sh
cd ./src/
python -m pytest tests/
``` -->

## Datasets

You can get datasets from [keel](https://sci2s.ugr.es/keel/imbalanced.php?order=name#sub10).

Here is the list of datasets that were used in this project:

|Name| classes | features  | instances | IR |
|---|---|---|---|---|
| ecoli1 | 2 |  7 | 336 | 3.36 |
| glass0 | 2 |  7 | 214 | 2.06 |

*TODO complete this table*
