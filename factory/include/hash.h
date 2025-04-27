#ifndef HASH_H
#define HASH_H

#define HASH_ATX_H1_MARKER 268
#define HASH_ATX_H2_MARKER 2978
#define HASH_ATX_HEADING 1719
#define HASH_DOCUMENT 2736
#define HASH_EMPH_EM 987
#define HASH_EMPH_STRONG 3762
#define HASH_EMPHASIS 977
#define HASH_EMPHASIS_DELIMITER 2402
#define HASH_FULL_REFERENCE_LINK 1179
#define HASH_HEADING_CONTENT 2457
#define HASH_INLINE 2375
#define HASH_INLINE_LINK 2580
#define HASH_LINK 4088
#define HASH_LINK_DESTINATION 378
#define HASH_LINK_LABEL 3844
#define HASH_LINK_REFERENCE_DESTINATION 3311
#define HASH_LINK_TEXT 1235
#define HASH_PARAGRAPH 3406
#define HASH_SECTION 3768
#define HASH_SHORTCUT_LINK 1841
#define HASH_TEXT 1057
#define HASH_URI_AUTOLINK 2830

/**
 * Compute the hash of the given string. It makes it easy to do switch-case
 * comparisons on string values.
 *
 * Taken from: https://stackoverflow.com/a/37121071
 */
const unsigned int hash(const char *str);

#endif
