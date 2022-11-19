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

    cell_hover = {  # for row hover use <tr> instead of <td>
        "selector": "td:hover",
        "props": [("background-color", "#D0DFFA")],
    }
    index_names = {
        "selector": ".index_name",
        "props": "font-style: italic; color: darkgrey; font-weight:normal;",
    }
    headers = {
        "selector": "th:not(.index_name)",
        "props": "background-color: #D0DFFA; color: black;",
    }

    table = table.set_table_styles([cell_hover, index_names, headers])

    table = table.set_table_styles(
        [
            {"selector": "th.col_heading", "props": "text-align: center;"},
            {"selector": "th.col_heading.level0", "props": "font-size: 1.5em;"},
            {"selector": "td", "props": "text-align: center; " "font-size: 1.2em;"},
            {"selector": "", "props": [("border", "2px black solid !important")]},
            {"selector": "tbody td", "props": [("border", "1px solid grey")]},
            {"selector": "th", "props": [("border", "1px solid grey")]},
        ],
        overwrite=False,
    )

    return table

