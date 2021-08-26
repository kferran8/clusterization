import streamlit as st
import pandas as pd
import cluster as cl
import base64
from io import BytesIO
import plotly.graph_objs as go
#Как развернуть приложение можно почитать здесь
#https://www.analyticsvidhya.com/blog/2021/06/deploy-your-ml-dl-streamlit-application-on-heroku/


# запись результата в эксель
def write_to_excel(df_initial, df_new, df_cluster_for_plot, df_agg):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    with writer as writer:
        df_initial.to_excel(writer, sheet_name='Исходная статистика', index=False)
        df_new.to_excel(writer, sheet_name='Кластеризация', index=False)
        df_cluster_for_plot.to_excel(writer, sheet_name='Для графика', index=False)
        df_agg.to_excel(writer, sheet_name='Групповые результаты', index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Функция позволяющая сделать download на выгрузку данных расчета
def get_table_download_link(df_initial, df_new, df_cluster_for_plot, df_agg):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = write_to_excel(df_initial, df_new, df_cluster_for_plot, df_agg)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Result_cluster.xlsx">' \
           f'Скачать xlsx файл результата</a>'  # decode b'abc' => abc


def app():
    st.title('Кластеризация')
    uploaded_file = st.file_uploader('Загрузите или перетащите файл Excel')
    try:
        df_initial = pd.read_excel(uploaded_file)
        st.subheader('Загруженные данные')
        st.write(df_initial)
    except Exception:
        st.subheader('Файл еще не загружен.')
    else:
        try:
            st.sidebar.subheader('Ввод количества кластеров')
            n_cluster = int(st.sidebar.number_input('Поле ввода количества кластеров',value=5))
            cluster = cl.Cluster(df_initial=df_initial, n_cluster= n_cluster, file_name ='Результаты кластеризации')
            cluster.fun()
            df_new = cluster.df_new
            st.subheader('Результат кластеризации')
            st.write(df_new)
            st.subheader('Графическая интерпритация')
            df_plot = cluster.df_cluster_for_plot


            st.sidebar.subheader('Настройки графика') # Боковая панель
            age_fig1 = st.sidebar.slider("Размер маркера 1: ", min_value=1,
                                   max_value=15, value=3, step=1)
            age_fig2 = st.sidebar.slider("Размер маркера 2: ", min_value=1,
                                   max_value=15, value=3, step=1)

            fig = go.Figure(data=go.Scatter(x=df_plot.iloc[:,0],
                                            y=df_plot.iloc[:,1],
                                            mode='markers',
                                            marker=dict(color='#008080', size=age_fig1),
                                            # line=dict(color='#00AF64'), opacity=0.8,
                                            name='Исходные данные'))

            fig.add_trace(go.Scatter(x=df_plot.iloc[:,2], y=df_plot.iloc[:,3],
                                     # line=dict(color='#CF2817'),
                                     mode='markers',
                                     marker=dict(color='#ff0000', size=age_fig2, symbol = 'circle'),
                                     name='Результаты кластеризации'))

            fig.update_layout(legend_orientation="h",
                              legend=dict(x=.5, xanchor="center"),
                              # title="Plot Title",
                              xaxis_title=df_plot.columns[0],
                              yaxis_title=df_plot.columns[1],
                              margin=dict(l=20, r=0, t=0, b=0))
            st.write(fig)

            # Загрузка данных результата
            st.subheader('Скачайте файл результата в Excel')
            st.markdown(get_table_download_link(df_initial=df_initial,
                                                df_new=df_new,
                                                df_cluster_for_plot=df_plot,
                                                df_agg=cluster.df_agg), unsafe_allow_html=True)
            st.caption('Автор: Капанский А.А., 2021 г.')
        except Exception as err:
            st.subheader('Что-то пошло не так. \n Ошибка:')
            st.subheader(err)
            # st.subheader('Некоректно введены данные!')
            # st.subheader('Исходный Excel файл должен содержать не больше двух ЧИСЛОВЫХ переменных (столбцов).')


if __name__ == '__main__':
    app()




