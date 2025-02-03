#ifndef HASH_H
#define HASH_H

#define HASH_ATX_H1_MARKER 11484553033317560587
#define HASH_ATX_HEADING 13866777968361919073
#define HASH_DOCUMENT 7572293989184612
#define HASH_EMPHASIS 7572334517618623
#define HASH_EMPHASIS_DELIMITER 11736955094843475965
#define HASH_HEADING_CONTENT 7588163596356272111
#define HASH_INLINE 6953632808036
#define HASH_PARAGRAPH 249902000470958555
#define HASH_SECTION 229482434843354

/**
 * Compute the hash of the given string. It makes it easy to do switch-case
 * comparisons on string values.
 *
 * Taken from: https://stackoverflow.com/a/37121071
 */
const unsigned long hash(const char *str);

#endif
