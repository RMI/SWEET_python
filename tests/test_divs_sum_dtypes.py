"""``DivsDF.sum`` keeps its dtypes, and stops asking pandas for a removed keyword.

The four ``reindex(...).infer_objects(...).fillna(0)`` chains in ``DivsDF.sum``
used to pass ``copy=False``. That keyword has been ignored since pandas 3.0 made
Copy-on-Write unconditional -- ``_check_copy_deprecation`` warns and nothing else
consults it -- and it is slated for removal in pandas 4, at which point the call
raises ``TypeError``.

The removal is worse than a plain ``TypeError`` here, because every caller of
``DivsDF.sum`` wraps it in a bare ``except:`` whose fallback (``sum(divs_df.values())``)
does not work on a pydantic model. The real failure surfaces as
``AttributeError: 'DivsDF' object has no attribute 'values'``, pointing nowhere
near the cause -- so ``test_removal_of_the_keyword_is_not_masked`` pins that too.

``infer_objects()`` itself stays. It does nothing for the reindex (added columns
are float64 NaN regardless of source dtype), but it is what keeps an object-dtype
input column from surviving ``fillna(0)`` and dragging the sum to object.
"""

import warnings

import pandas as pd
import pytest

from SWEET_python.class_defs import DivsDF, LandfillWasteMassDF


WASTE_TYPES = ["food", "green", "wood", "paper_cardboard", "plastic"]
YEARS = [2000, 2001, 2002]


def _frame(value, columns=WASTE_TYPES):
    return pd.DataFrame({c: [float(value)] * len(YEARS) for c in columns}, index=YEARS)


def _divs(**overrides):
    frames = {
        "compost": _frame(1),
        "anaerobic": _frame(2),
        "combustion": _frame(3),
        "recycling": _frame(4),
    }
    frames.update(overrides)
    return DivsDF(**frames)


class TestNoDeprecatedKeyword:
    def test_sum_emits_no_pandas_deprecation(self):
        """The 96 Pandas4Warnings this suite used to raise, pinned at zero."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _divs().sum()

        pandas_deprecations = [
            w for w in caught if isinstance(w.message, pd.errors.PandasChangeWarning)
        ]
        assert pandas_deprecations == [], [str(w.message) for w in pandas_deprecations]

    def test_removal_of_the_keyword_is_not_masked(self, monkeypatch):
        """With the keyword gone, sum() must still work rather than fail obliquely.

        Simulates pandas 4 by making ``infer_objects`` reject every argument.
        Before the fix this raised ``TypeError`` inside ``sum()``, which the
        caller's bare ``except:`` turned into an unrelated ``AttributeError``.
        """
        original = pd.DataFrame.infer_objects

        def no_keywords_accepted(self, *args, **kwargs):
            if args or kwargs:
                raise TypeError(
                    "infer_objects() takes 1 positional argument but 2 were given"
                )
            return original(self)

        monkeypatch.setattr(pd.DataFrame, "infer_objects", no_keywords_accepted)

        divs = _divs()
        assert divs.sum().loc[2000, "food"] == pytest.approx(10.0)

        # And through the real caller, whose bare `except:` did the masking.
        landfilled = LandfillWasteMassDF.create(
            waste_generated_df=_frame(100),
            divs_df=divs,
            fraction_of_waste=0.5,
            waste_types=WASTE_TYPES,
        )
        assert landfilled.df.loc[2000, "food"] == pytest.approx(45.0)


class TestDtypesAndValues:
    def test_float_inputs_sum_to_float(self):
        summed = _divs().sum()

        assert list(summed.columns) == WASTE_TYPES
        assert (summed.dtypes == "float64").all(), summed.dtypes.to_dict()
        assert (summed == 10.0).all().all()

    def test_reindex_fills_missing_columns_with_zero_as_float(self):
        """A stream missing a waste type contributes 0 to it, not NaN or object."""
        summed = _divs(compost=_frame(1, columns=["food", "green"])).sum()

        assert list(summed.columns) == sorted(WASTE_TYPES)
        assert (summed.dtypes == "float64").all(), summed.dtypes.to_dict()
        assert summed.loc[2000, "food"] == pytest.approx(10.0)
        # compost sat out: 2 + 3 + 4, no NaN leaking through.
        assert summed.loc[2000, "plastic"] == pytest.approx(9.0)

    def test_object_dtype_input_is_inferred_back_to_float(self):
        """Why infer_objects() stays: a boxed column must not survive fillna(0)."""
        boxed = _frame(1)
        boxed["food"] = pd.Series([1.0] * len(YEARS), index=YEARS, dtype=object)
        assert boxed.dtypes["food"] == object

        summed = _divs(compost=boxed).sum()

        assert summed.dtypes["food"] == "float64", summed.dtypes.to_dict()
        assert summed.loc[2000, "food"] == pytest.approx(10.0)
