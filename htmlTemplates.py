css = '''
<style>

.chat-message.user {
    border: 1px solid black;
    margin-left: 20rem;
    background-color: #89CFF0;
    overflow: scroll;

}
.chat-message.bot {
    border: 1px solid black;
    margin-right: 20rem;
    background-color: #B9D9EB;
    overflow: scroll;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-top: 1rem; display: flex
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color:#000000;
}


'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.vectorstock.com/i/1000x1000/74/57/ai-robot-flat-color-icon-vector-29147457.webp" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://thumbs.dreamstime.com/z/businessman-profile-icon-male-portrait-flat-design-vector-illustration-47075259.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''










