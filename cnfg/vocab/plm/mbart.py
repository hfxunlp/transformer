#encoding: utf-8

pad_id, sos_id, eos_id, unk_id, mask_id = 1, 0, 2, 3, 250026#250053
vocab_size = 250027#250054
pemb_start_ind = 2

shift_target_lang_id = True#False
add_sos_id = None

lang_id = {"ar_AR": 250001, "cs_CZ": 250002, "de_DE": 250003, "en_XX": 250004, "es_XX": 250005, "et_EE": 250006, "fi_FI": 250007, "fr_XX": 250008, "gu_IN": 250009, "hi_IN": 250010, "it_IT": 250011, "ja_XX": 250012, "kk_KZ": 250013, "ko_KR": 250014, "lt_LT": 250015, "lv_LV": 250016, "my_MM": 250017, "ne_NP": 250018, "nl_XX": 250019, "ro_RO": 250020, "ru_RU": 250021, "si_LK": 250022, "tr_TR": 250023, "vi_VN": 250024, "zh_CN": 250025}
#lang_id = {"ar_AR": 250001, "cs_CZ": 250002, "de_DE": 250003, "en_XX": 250004, "es_XX": 250005, "et_EE": 250006, "fi_FI": 250007, "fr_XX": 250008, "gu_IN": 250009, "hi_IN": 250010, "it_IT": 250011, "ja_XX": 250012, "kk_KZ": 250013, "ko_KR": 250014, "lt_LT": 250015, "lv_LV": 250016, "my_MM": 250017, "ne_NP": 250018, "nl_XX": 250019, "ro_RO": 250020, "ru_RU": 250021, "si_LK": 250022, "tr_TR": 250023, "vi_VN": 250024, "zh_CN": 250025, "af_ZA": 250026, "az_AZ": 250027, "bn_IN": 250028, "fa_IR": 250029, "he_IL": 250030, "hr_HR": 250031, "id_ID": 250032, "ka_GE": 250033, "km_KH": 250034, "mk_MK": 250035, "ml_IN": 250036, "mn_MN": 250037, "mr_IN": 250038, "pl_PL": 250039, "ps_AF": 250040, "pt_XX": 250041, "sv_SE": 250042, "sw_KE": 250043, "ta_IN": 250044, "te_IN": 250045, "th_TH": 250046, "tl_XX": 250047, "uk_UA": 250048, "ur_PK": 250049, "xh_ZA": 250050, "gl_ES": 250051, "sl_SI": 250052}
