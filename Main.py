import streamlit as st

def run():
    st.markdown("# Nordic Nature & Nitrogen")
    st.sidebar.markdown("# Introduction")
    main_text()
    st.session_state.autoload = True 
    
    
def main_text():
    multi = '''
    This project revolves around the inclusion of the bidirectional flux process in air pollution models in Denmark, Norway and Sweden. \
        The bidirectional flux is a way to describe the fact that the vegetation play an active role in the dynamics of ammonia in the \
        atmosphere, and this role should be described in the best possible way in the air pollution models. \
            
    This project constitutes the second of three contig-uous projects with respect to the assessment of the impact of nitrogen on Nordic nature. 

    During this project the following main activities were carried out:
    
    * Completion of the implementation of the bidirectional flux parameterisation in the DEHM, EMEP and MATCH models
    * Evaluation of the performance of the new and old versions of the models with measurements
    * First simulations with the new parameterisation of dry depostion of ammonia
    * First investigation of the impact of including the process in the models.

    '''
    
    
    st.markdown(multi)


if __name__ == '__main__':
    run()