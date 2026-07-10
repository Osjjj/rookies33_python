import streamlit as st

st.title("Hello Streamlit-er 👋")
st.markdown(
    """ 
    This is a playground for you to try Streamlit and have fun. 

    **There's :rainbow[so much] you can build!**
    
    We prepared a few examples for you to get started. Just 
    click on the buttons above and discover what you can do 
    with Streamlit. 
    """
)

st.sidebar.success('메뉴 선택')

st.code('num=100')

name = st.text_input('이름')
print(name)

if st.button("Send balloons!"):
    st.write('버튼을 클릭했어요!')
    print('버튼클릭')
    st.balloons()
