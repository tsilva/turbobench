from xml.etree import ElementTree

import pytest

from turbobench.readme_chart import (
    export_readme_chart,
    full_publication_chart,
    peak_rows,
    readme_chart,
)
from turbobench.stats import paired_statistics


def result_for(values):
    return {
        "comparison": {
            "left": {"provider": "upstream<&", "version": "1"},
            "right": {"provider": "candidate", "version": "2"},
            "shapes": {
                str(2**i): {
                    "statistics": paired_statistics(
                        [{"left_sps": [100] * 3, "right_sps": [sps] * 3}] * 7
                    )
                }
                for i, sps in enumerate(values)
            },
        }
    }


@pytest.mark.parametrize(
    "values,shown,omitted",
    [
        ([200, 400, 300, 100], [1, 2], [4, 8]),
        ([200, 400, 400, 300], [1, 2], [4, 8]),
        ([200, 150, 400, 300], [1, 2, 4], [8]),
        ([200, 400, 600], [1, 2, 4], []),
        ([200], [1], []),
    ],
)
def test_peak_view_keeps_complete_prefix_including_recovery(values, shown, omitted):
    rows, dropped = peak_rows(result_for(values))
    assert [int(shape) for shape, _ in rows] == shown
    assert dropped == omitted


def test_readme_view_uses_one_scale_and_keeps_speedup():
    chart = readme_chart(result_for([200, 400, 300]), diagnostic=False)
    svg = ElementTree.fromstring(chart)
    bars = [el for el in svg.iter() if "data-provider" in el.attrib]
    scales = [float(el.attrib["height"]) / float(el.attrib["data-sps"]) for el in bars]
    assert scales == pytest.approx([scales[0]] * 4)
    baselines = [float(el.attrib["y"]) + float(el.attrib["height"]) for el in bars]
    assert baselines == pytest.approx([baselines[0]] * 4)
    assert float(bars[0].attrib["x"]) + float(bars[0].attrib["width"]) < float(bars[1].attrib["x"])
    assert int(svg.attrib["height"]) <= 450
    assert "4.00\u00d7" in chart
    assert "95% paired CI" not in chart
    assert "Later counts omitted here:" not in chart
    assert "upstream&lt;&amp;" in chart
    assert "DIAGNOSTIC" not in chart
    assert 'width="800"' in chart


def test_export_fails_before_writing_for_invalid_proof_or_inside_proof(tmp_path):
    output = tmp_path / "chart.svg"
    with pytest.raises(ValueError):
        export_readme_chart(tmp_path / "missing", output)
    assert not output.exists()
    with pytest.raises(ValueError, match="outside the immutable"):
        export_readme_chart(tmp_path, output)
    assert not output.exists()


def test_smoke_readme_chart_does_not_invent_confidence_interval():
    result = result_for([200])
    result["comparison"]["shapes"]["1"]["statistics"]["bootstrap"] = None
    chart = readme_chart(result, diagnostic=True)
    assert "DIAGNOSTIC" in chart
    assert "One sample; no confidence interval" not in chart
    assert "95% paired CI" not in chart


@pytest.mark.parametrize("renderer", [readme_chart, full_publication_chart])
def test_publication_labels_round_sps_without_changing_bar_data(renderer):
    result = result_for([69797.5, 124814.6])
    result["comparison"]["shapes"]["1"]["statistics"]["median_left_sps"] = 192.6
    svg = ElementTree.fromstring(renderer(result, diagnostic=False))
    text = " ".join(svg.itertext())
    assert "69,798" in text and "124,815" in text and "193" in text
    assert "69,797.5" not in text and "124,814.6" not in text and "192.6" not in text
    assert "95% paired CI" not in text and "Bars:" not in text
    values = [float(el.attrib["data-sps"]) for el in svg.iter() if "data-sps" in el.attrib]
    assert 192.6 in values and 69797.5 in values
