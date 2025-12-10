# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.17.0",
#     "pandas>=2.3.3",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Choice a number
    """)
    return


@app.cell
def _(mo):
    number = mo.ui.slider(1, 10)
    return (number,)


@app.cell
def _(number):
    number
    return


@app.cell
def _(number):
    number.value
    return


if __name__ == "__main__":
    app.run()
