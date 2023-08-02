"""AtomsViewer
Important: AtomsViewer only works in Jupyter notebooks!
"""

import json
import os
import time
from typing import Dict, Optional, List, Union, Any

import numpy.typing as npt

_SCRIPT_TAG_OPEN = "<script>"
_SCRIPT_TAG_CLOSE = "</script>"


def _vec3d_to_json(v: npt.NDArray[Any]) -> Dict[str, float]:
    assert v.shape == (3,)
    c = v.astype(float)
    return {"x": c[0], "y": c[1], "z": c[2]}


def _to_json(o: Any) -> str:
    return json.dumps(o)


def _dict_to_style(d: Dict[str, Any]) -> str:
    return ";".join([str(k) + ":" + str(v) for k, v in d.items()])


def _display_html(html: str) -> None:
    # Import here to avoid issues with CI
    import IPython.display

    IPython.display.publish_display_data({"text/html": html})


def _display_js(script: str) -> None:
    # Import here to avoid issues with CI
    import IPython.display

    IPython.display.display_javascript(script, raw=True)


def init() -> None:
    # Display loading banner
    loading_div_class = "atoms_viewer_loading"
    html = f"<div class={loading_div_class}>Loading atoms_viewer module. Please wait.</div>"
    _display_html(html)

    # Load library
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, "atoms_viewer.js"), mode="r") as f:
        file_content = f.read()

    # Library has to be loaded through <script> tags
    html = _SCRIPT_TAG_OPEN + file_content + _SCRIPT_TAG_CLOSE
    _display_html(html)

    # Verify that library loaded correctly
    update_script = (
        f"change_element_of_class({_to_json(loading_div_class)}, "
        '"<p>Success: <code>atoms_viewer</code> module is loaded!</p>");'
    )
    _display_js(update_script)


class AtomsViewer:
    _default_style_config: Dict[str, str] = {
        "width": "500px",
        "height": "500px",
        "position": "relative",
    }

    _default_create_config: Dict[str, Union[str, int, float]] = {
        "backgroundColor": "white",
        "backgroundAlpha": 0.0,
    }

    _open_promise = "$3Dmolpromise.then(function() {"
    _close_promise = "});"

    _default_arrow_config: Dict[str, Union[str, int, float]] = {
        "color": "black",
        "alpha": 0.9,
        "radius": 0.1,
        "mid": 0.75,
    }

    def __init__(self) -> None:
        self.instance_id = str(time.time()).replace(".", "")
        self.elements: List[str] = []
        self.statements: List[str] = []

        self.created = False

    def __del__(self) -> None:
        if not self.created:
            return

        self.elements.clear()
        self.statements = [f"delete_viewer({_to_json(self.instance_id)});"]
        self.process(dry_run=False, debug=False)

    def create(
        self, style: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None
    ) -> "AtomsViewer":
        if self.created:
            raise RuntimeError("Cannot call create() twice.")

        # Div style
        style_dict = self._default_style_config.copy()
        if style is not None:
            style_dict.update(style)

        element = f"<div id={_to_json(self.instance_id)} style={_to_json(_dict_to_style(style_dict))}></div>"
        self.elements.append(element)

        # Viewer style
        create_dict = self._default_create_config.copy()
        if config is not None:
            create_dict.update(config)

        statement = f"create_viewer({_to_json(self.instance_id)}, {_to_json(create_dict)});"
        self.statements.append(statement)

        self.created = True
        return self

    def queue(self, name: str, *args: Any) -> "AtomsViewer":
        if not self.created:
            raise RuntimeError("Call create() method first.")

        statement = (
            f"viewer_dict[{_to_json(self.instance_id)}].{name}"
            f"(" + ",".join(_to_json(arg) for arg in args) + ");"
        )
        self.statements.append(statement)
        return self

    def remove_all(self) -> "AtomsViewer":
        self.queue("removeAllLabels").queue("removeAllModels").queue("removeAllShapes").queue(
            "removeAllSurfaces"
        )

        return self

    def process(self, dry_run: bool = False, debug: bool = False) -> Optional[str]:
        if not self.created:
            raise RuntimeError("Call create() method first.")

        code = "\n".join(
            self.elements
            + [_SCRIPT_TAG_OPEN, self._open_promise]
            + self.statements
            + [self._close_promise, _SCRIPT_TAG_CLOSE]
        )

        if not dry_run:
            _display_html(code)
            self.elements.clear()
            self.statements.clear()

        if debug:
            return code

        return None

    def add_arrows(
        self,
        positions: npt.NDArray[Any],
        vectors: npt.NDArray[Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> "AtomsViewer":
        assert positions.shape[1] == 3
        assert positions.shape == vectors.shape

        arrow_dict = self._default_arrow_config.copy()

        if config is not None:
            arrow_dict.update(config)

        for i in range(positions.shape[0]):
            self.queue(
                "addArrow",
                {
                    **arrow_dict,
                    "start": _vec3d_to_json(positions[i]),
                    "end": _vec3d_to_json(positions[i] + vectors[i]),
                },
            )

        return self

    def download_png(self, file_path: str) -> "AtomsViewer":
        statement = f"viewer_to_png({_to_json(self.instance_id)}, {_to_json(file_path)});"
        self.statements.append(statement)
        return self

    def take_snapshot(self) -> "AtomsViewer":
        statement = f"push_to_buffer({_to_json(self.instance_id)});"
        self.statements.append(statement)
        return self

    def download_apng(self, file_path: str, delay: float = 1000) -> "AtomsViewer":
        statement = f"process_buffer({_to_json(self.instance_id)}, {_to_json(delay)}, {_to_json(file_path)});"
        self.statements.append(statement)
        return self
