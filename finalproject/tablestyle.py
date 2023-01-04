def make_clickable(val):

    """Make urls in dataframe clickable in html output"""

    if val is not None:
        if "," in val:
            x = val.split(",")
            return f'{x[0]} <a target="_blank" href="{x[1]}">plot</a>'
        else:
            return f'<a target="_blank" href="{val}">plot</a>'
    else:
        return


def table_style(table):

    """Applies style to a given table"""

    cell_hover = {
        "selector": "tr:hover",
        "props": [("background-color", "#92c5de")],
    }
    border_fix = {
        "selector": "td",
        "props": "border - collapse: collapse",
    }

    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #92c5de; color: black;",
    }
    borders = {"selector": "td, th", "props": "border: 1px solid #000000;padding: 6px;"}

    table = table.set_table_styles([border_fix])

    table = table.set_table_styles([cell_hover, index_names, headers, borders])

    table = table.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading", "props": "padding-top: 10px;"},
            {"selector": "th.col_heading", "props": "padding-bottom: 10px"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.0em;"},
            {"selector": "td", "props": "text-align: center; " "font-size: 0.9em;"},
            {"selector": "", "props": [("border", "2px black solid")]},
        ],
        overwrite=False,
    )
    table.hide(axis="index")

    return table
