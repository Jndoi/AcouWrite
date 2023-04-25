"""
@Project :symspelldemo
@File ：word_suggestion.py
@Date ： 2022/10/10 15:00
@Author ： Qiuyang Zeng
@Software ：PyCharm
https://symspellpy.readthedocs.io/en/latest/index.html
"""
import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)


def spell_correction(term):
    suggestion = sym_spell.lookup(
        term, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
    )[0].term
    return suggestion
