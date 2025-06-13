#include "hash.h"

// HASH_MODULO is useful to reduce the range of the generated hashes such that
// they can be use in switch-case statements (requires int).
#define HASH_MODULO 4093

const unsigned int hash(const char *str) {
  unsigned long hash = 5381;
  int c;

  while ((c = *str++))
    hash = ((hash << 5) + hash) + c;
  return (unsigned int)(hash % HASH_MODULO);
}
