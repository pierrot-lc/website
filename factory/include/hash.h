#ifndef HASH_H
#define HASH_H

#define HASH_CONTENT 229462175750528
#define HASH_HEADING 229468225748821
#define HASH_SECTION 229482434843354
#define HASH_SOURCE_FILE 13894110916739591349
#define HASH_TEXT 6385723658

/**
 * Compute the hash of the given string. It makes it easy to do switch-case
 * comparisons on string values.
 *
 * Taken from: https://stackoverflow.com/a/37121071
 */
const unsigned long hash(const char *str);

#endif
