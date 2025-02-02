#ifndef HASH_H
#define HASH_H

/**
 * Compute the hash of the given string. It makes it easy to do switch-case
 * comparisons on string values.
 *
 * Taken from: https://stackoverflow.com/a/37121071
 */
const unsigned long hash(const char *str);

#endif
