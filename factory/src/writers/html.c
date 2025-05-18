#include <stdio.h>

#include "tree.h"
#include "writers/article_metadata.h"
#include "writers/body_main.h"
#include "writers/head.h"
#include "writers/header.h"

void write_html(FILE *file, Node *tree) {
  fprintf(file, "<!DOCTYPE html>\n");
  fprintf(file, "<html>\n");

  write_head(file, tree);

  fprintf(file, "<body>\n");
  write_header(file, tree);
  fprintf(file, "<main>\n");
  write_article_metadata(file, tree);
  write_body_main(file, tree);
  fprintf(file, "</main>\n");
  fprintf(file, "</body>\n");

  fprintf(file, "</html>\n");
}
