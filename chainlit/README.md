## 🚀 Quickstart

```
$ chainlit run xxx.py
```
- After runing, Your app is available at http://localhost:8000

## Website

- 👉 ifame : [index](index.html)
- using python server as demo
  
    ```
    $ python -m http.server 8001
    ```

## 🗄️ Build a vector database

- using `chroma`: the open-source embedding database.
- [chroma github](https://github.com/chroma-core/chroma)
- https://www.trychroma.com/

```
$ python build_dataset/build_few_dataset.py --test_dataset_path data/few_dataset.json
```