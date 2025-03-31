#ifndef HASH_H
#define HASH_H

#define HASH_ATX_H1_MARKER 268
#define HASH_ATX_H2_MARKER 2978
#define HASH_ATX_HEADING 1719
#define HASH_DOCUMENT 2736
#define HASH_EMPHASIS 977
#define HASH_EMPHASIS_DELIMITER 2402
#define HASH_HEADING_CONTENT 2457
#define HASH_INLINE 2375
#define HASH_PARAGRAPH 3406
#define HASH_SECTION 3768

/**
 * Compute the hash of the given string. It makes it easy to do switch-case
 * comparisons on string values.
 *
 * Taken from: https://stackoverflow.com/a/37121071
 */
const unsigned int hash(const char *str);

#endif
