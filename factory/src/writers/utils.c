#include <stdio.h>

#include "writers/utils.h"

void write_link(FILE *file, const char *href, const char *content) {
  fprintf(file, "<a href=\"%s\">%s</a>", href, content);
}
