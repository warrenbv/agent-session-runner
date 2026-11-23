"""urllib behaviour shared by the two clients that send a credential: the control plane and GitHub.

urllib's default redirect handler follows a redirect and sends the Authorization header to the new
address. These clients never follow one. The redirect answer reaches the caller as an HTTP error.
"""

from urllib.request import HTTPRedirectHandler


class NoRedirects(HTTPRedirectHandler):

    def http_error_302(self, req, fp, code, msg, headers) -> None:
        return None

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302
