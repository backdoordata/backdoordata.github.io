---
layout: archive
permalink: /data-analysis-and-visualizations/
title: "Data Analysis & Visualizations"
author_profile: true
header:
    image: "/images/data-A&V-header.jpg"
---

{% include absolute_url %}
{% include group-by-array collection=site.posts field="tag_archive" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}