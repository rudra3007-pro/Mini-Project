# auth.py — Google OAuth blueprint
import os
from flask import Blueprint, redirect, url_for, session, jsonify
from authlib.integrations.flask_client import OAuth

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")
oauth   = OAuth()

def init_oauth(app):
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
    oauth.init_app(app)
    oauth.register(
        name="google",
        client_id=os.environ.get("GOOGLE_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"),
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

@auth_bp.route("/login")
def login():
    redirect_uri = url_for("auth.callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri)

@auth_bp.route("/callback")
def callback():
    token    = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo") or oauth.google.userinfo()
    session["user"] = {
        "name":    userinfo.get("name"),
        "email":   userinfo.get("email"),
        "picture": userinfo.get("picture"),
    }
    return redirect("/")

@auth_bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

@auth_bp.route("/me")
def me():
    user = session.get("user")
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(user)