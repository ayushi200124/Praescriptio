from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length,Email, EqualTo, email_validator
class RegistrationForm(FlaskForm):
    username= StringField('Username', validators=[DataRequired(),Length(min=3,max=20)])
    email= StringField('Email',validators=[DataRequired(),Email()])
    password= PasswordField('Password',validators=[DataRequired()])
    confirm_password= PasswordField('Confirm Password',validators=[DataRequired(), EqualTo('password')])
    submit= SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email= StringField('Email',validators=[DataRequired(),Email()])
    password= PasswordField('Password',validators=[DataRequired()])
    remember= BooleanField('Remember Me')
    submit= SubmitField('Login')

class ContactForm(FlaskForm):
    name= StringField('Username', validators=[DataRequired(),Length(min=3,max=20)])
    email= StringField('Email',validators=[DataRequired(),Email()])
    subject= StringField('Username', validators=[DataRequired(),Length(min=5,max=200)])
    description= StringField('Username', validators=[DataRequired(),Length(min=5,max=500)])
    submit= SubmitField('Contact')	
