"""The app's OpenAPI spec, as served and as snapshotted for the frontend client.

* :mod:`~vtsearch.openapi.postprocess` - what the spec needs beyond what
  flask-smorest writes: an ``operationId`` per operation, from its view
  function's name, so the generated TypeScript client's method names do not
  churn with URLs.
"""
