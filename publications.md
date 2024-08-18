---
title: Papers
---
# Papers

{% for item in site.data.publications %}
- [{{ item.name }}]({{ item.link }}) - *{{ item.publisher }}*
{% endfor %}
