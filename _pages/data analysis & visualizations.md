---
layout: archive
permalink: /analyzing-data/
title: "Analyzing Data"
author_profile: true
breadcrumbs: true
header:
    image: "/images/data-A&V-header.jpg"
---

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}