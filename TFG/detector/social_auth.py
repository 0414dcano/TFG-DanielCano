from allauth.socialaccount.models import SocialApp
from django.contrib.sites.models import Site

def setup_google_login():
    site = Site.objects.get_current()
    
    app, created = SocialApp.objects.get_or_create(
        provider='google',
        name='Google Login',
        client_id='320396564843-0i62jv1l98ibhnqdavm3mrtj9r99b95p.apps.googleusercontent.com',
        secret='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',
    )
    app.sites.add(site)
