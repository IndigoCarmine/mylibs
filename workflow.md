# 開発時のコマンド

依存を入れる（`uv.lock` に固定されたものが入り、このプロジェクト自身も editable で入る）。

```bash
uv sync
```

CI (`.github/workflows/main.yml`) が回しているのと同じチェック。

```bash
# テスト。-t . は tests/mole が mole パッケージを shadow しないために必要。
uv run python -m unittest discover -s tests -t .

# 型チェック。
uv run mypy src/

# バージョン互換チェック。
# DefaultFiles のスクリプトはクラスタ側の Python 3.6.8 で実行されるので別ターゲット。
uv run vermin --no-tips --target=3.6 --violations src/gromacs/DefaultFiles/
uv run vermin --no-tips --target=3.12 --violations --exclude-regex '.*DefaultFiles.*' src/
```

`.pyi` スタブを作り直す。

```bash
uv run stubgen -p mole -p gromacs -p base_utils -o src/
```

リリース時はバージョンを上げる。`setup.py` は無く、`pyproject.toml` の `[project] version` が正。

```bash
uv build
```
