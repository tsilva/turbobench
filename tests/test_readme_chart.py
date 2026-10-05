from xml.etree import ElementTree

import pytest

from turbobench.readme_chart import export_readme_chart, peak_rows, readme_chart
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


def test_readme_view_uses_one_scale_and_shape_local_paired_uncertainty():
    chart = readme_chart(result_for([200, 400, 300]), diagnostic=False)
    svg = ElementTree.fromstring(chart)
    bars = [el for el in svg.iter() if "data-provider" in el.attrib]
    scales = [float(el.attrib["width"]) / float(el.attrib["data-sps"]) for el in bars]
    assert scales == pytest.approx([scales[0]] * 4)
    assert all(float(el.attrib["x"]) == 36 for el in bars)
    assert "95% paired CI: 4.00\u20134.00\u00d7" in chart
    assert "Later counts omitted here: 4" in chart
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
    assert "One sample; no confidence interval" in chart
    assert "95% paired CI" not in chart
