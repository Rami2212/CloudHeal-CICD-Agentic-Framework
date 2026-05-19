---
dataset_info:
- config_name: default
  features:
  - name: language
    dtype: string
  - name: id
    dtype: int64
  - name: repo_owner
    dtype: string
  - name: repo_name
    dtype: string
  - name: head_branch
    dtype: string
  - name: workflow_name
    dtype: string
  - name: workflow_filename
    dtype: string
  - name: workflow_path
    dtype: string
  - name: contributor
    dtype: string
  - name: sha_fail
    dtype: string
  - name: sha_success
    dtype: string
  - name: workflow
    dtype: string
  - name: logs
    list:
    - name: step_name
      dtype: string
    - name: log
      dtype: string
  - name: diff
    dtype: string
  - name: difficulty
    dtype: int64
  - name: changed_files
    sequence: string
  - name: commit_link
    dtype: string
  - name: commit_date
    dtype: string
  splits:
  - name: test
    num_bytes: 11705172.916666666
    num_examples: 68
  download_size: 1450880
  dataset_size: 11705172.916666666
- config_name: old
  features:
  - name: language
    dtype: string
  - name: id
    dtype: int64
  - name: repo_owner
    dtype: string
  - name: repo_name
    dtype: string
  - name: head_branch
    dtype: string
  - name: workflow_name
    dtype: string
  - name: workflow_filename
    dtype: string
  - name: workflow_path
    dtype: string
  - name: contributor
    dtype: string
  - name: sha_fail
    dtype: string
  - name: sha_success
    dtype: string
  - name: workflow
    dtype: string
  - name: logs
    list:
    - name: step_name
      dtype: string
    - name: log
      dtype: string
  - name: diff
    dtype: string
  - name: difficulty
    dtype: int64
  - name: changed_files
    sequence: string
  - name: commit_link
    dtype: string
  - name: commit_date
    dtype: string
  splits:
  - name: test
    num_bytes: 24787425
    num_examples: 144
  download_size: 3550754
  dataset_size: 24787425
configs:
- config_name: default
  data_files:
  - split: test
    path: data/python/test-*
- config_name: old
  data_files:
  - split: test
    path: old/python/test-*
---

# 🏟️ Long Code Arena (CI builds repair)

This is the benchmark for CI builds repair task as part of the
🏟️ [Long Code Arena benchmark](https://huggingface.co/spaces/JetBrains-Research/long-code-arena).

🛠️ Task. Given the logs of a failed GitHub Actions workflow and the corresponding repository snapshot, 
repair the repository contents in order to make the workflow pass.

All the data is collected from repositories published under permissive licenses (MIT, Apache-2.0, BSD-3-Clause, and BSD-2-Clause). The datapoints can be removed upon request.

To score your model on this dataset, you can use [**CI build repair benchmark**](https://github.com/JetBrains-Research/lca-baselines/tree/main/ci-builds-repair/ci-builds-repair-benchmark).
📩 If you have any questions or requests concerning this dataset, please contact lca@jetbrains.com

## How-to

### List all the available configs
   via [`datasets.get_dataset_config_names`](https://huggingface.co/docs/datasets/v2.14.3/en/package_reference/loading_methods#datasets.get_dataset_config_names)
   and choose an appropriate one.

   Current configs: `python`

### Load the data
   via [`load_dataset`](https://huggingface.co/docs/datasets/v2.14.3/en/package_reference/loading_methods#datasets.load_dataset):

    from datasets import load_dataset

    dataset = load_dataset("JetBrains-Research/lca-ci-builds-repair", split="test")

   Note that all the data we have is considered to be in the test split.  
   **NOTE**: If you encounter any errors with loading the dataset on Windows, update the `datasets` library (was tested on `datasets==2.16.1`)

### Usage

For the dataset usage please refer to our [CI builds repair benchmark](https://github.com/JetBrains-Research/lca-baselines/tree/main/ci-builds-repair/ci-builds-repair-benchmark).
Its workflow is following:

1. Repairs repo by fix_repo_function function that utilizes repo state and logs of fails;
2. Sends the datapoints to GitHub to run workflows;
3. Requests results from GitHub;
4. Analyzes results and prints them;
5. Clones the necessary repos to the user's local machine.

  The user should run their model to repair the failing CI workflows, and the benchmark will push commits to GitHub,
  returning the results of the workflow runs for all the datapoints.


## Dataset Structure

This dataset contains logs of the failed GitHub Action workflows for some commits
followed by the commit that passes the workflow successfully.

Note that, unlike other 🏟️ Long Code Arena datasets, this dataset does not contain repositories.


### Datapoint Schema


Each example has the following fields:

| Field               | Description                                                                                                                  |
|---------------------|------------------------------------------------------------------------------------------------------------------------------|
| `contributor`       | Username of the contributor that committed changes                                                                           |
| `difficulty`        | Difficulty of the problem (assessor-based. 1 means that the repair requires only the code formatting)                                                                                   |
| `diff`              | Contents of the diff between the failed and the successful commits                                                           |
| `head_branch`       | Name of the original branch that the commit was pushed at                                                                    |
| `id`                | Unique ID of the datapoint                                                                                                   |
| `language`          | Main language of the repository                                                                                                    |
| `logs`              | List of dicts with keys `log` (logs of the failed job, particular step) and `step_name` (name of the failed step of the job) |
| `repo_name`         | Name of the original repository (second part of the `owner/name` on GitHub)                                                        |
| `repo owner`        | Owner of the original repository (first part of the `owner/name` on GitHub)                                                        |
| `sha_fail`          | SHA of the failed commit                                                                                                     |
| `sha_success`       | SHA of the successful commit                                                                                                 |
| `workflow`          | Contents of the workflow file                                                                                                |
| `workflow_filename` | The name of the workflow file (without directories)                                                                          |
| `workflow_name`     | The name of the workflow                                                                                                     |
| `workflow_path`     | The full path to the workflow file     
| `changed_files`     | List of files changed in diff
| `commit_link`       | URL to commit corresponding to failed job

### Datapoint Example


```
{'contributor': 'Gallaecio',
 'diff': 'diff --git a/scrapy/crawler.py b/scrapy/crawler.py/n<...>',
 'difficulty': '2',
 'head_branch': 'component-getters',
 'id': 18,
 'language': 'Python',
 'logs': [{'log': '##[group]Run pip install -U tox\n<...>',
           'step_name': 'checks (3.12, pylint)/4_Run check.txt'}],
 'repo_name': 'scrapy',
 'repo_owner': 'scrapy',
 'sha_fail': '0f71221cf9875ed8ef3400e1008408e79b6691e6',
 'sha_success': 'c1ba9ccdf916b89d875628ba143dc5c9f6977430',
 'workflow': 'name: Checks\non: [push, pull_request]\n\n<...>',
 'workflow_filename': 'checks.yml',
 'workflow_name': 'Checks',
 'workflow_path': '.github/workflows/checks.yml',
 'changed_files': ["scrapy/crawler.py"],
 'commit_link': "https://github.com/scrapy/scrapy/tree/0f71221cf9875ed8ef3400e1008408e79b6691e6"}
```

## Citing
```
@article{bogomolov2024long,
  title={Long Code Arena: a Set of Benchmarks for Long-Context Code Models},
  author={Bogomolov, Egor and Eliseeva, Aleksandra and Galimzyanov, Timur and Glukhov, Evgeniy and Shapkin, Anton and Tigina, Maria and Golubev, Yaroslav and Kovrigin, Alexander and van Deursen, Arie and Izadi, Maliheh and Bryksin, Timofey},
  journal={arXiv preprint arXiv:2406.11612},
  year={2024}
}
```
You can find the paper [here](https://arxiv.org/abs/2406.11612).
