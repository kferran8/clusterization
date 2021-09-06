import sys

import streamlit as st
import pandas as pd
import cluster as cl
import numpy as np
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
@st.cache
def get_table_download_link(df_initial, df_new, df_cluster_for_plot, df_agg):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = write_to_excel(df_initial, df_new, df_cluster_for_plot, df_agg)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="Result_cluster.xlsx">' \
           f'Скачать xlsx файл результата</a>'  # decode b'abc' => abc

# Кэшируем функцию создания кластеризатора
@st.cache(allow_output_mutation=True)
def create_cluster(df_initial, n_cluster):
    cluster = cl.Cluster(df_initial=df_initial, n_cluster=n_cluster)
    cluster.fun()
    return cluster


def app():
    st.title('Multiparameter clustering')
    uploaded_file = st.file_uploader('Загрузите или перетащите файл Excel')
    try:
        # Читаем вель эксель файл
        xls = pd.ExcelFile(uploaded_file)
    except Exception as err:
        st.subheader('Excel файл еще не загружен.')
        st.text(err)
    else:
        try:
            st.sidebar.title('Настройки')
            # Смотрим какие есть листы и загружаем в селект бокс
            st.sidebar.subheader('Параметры рабочего листа файла')
            selected_sheet_names = st.sidebar.selectbox('Выберите лист загруженного файла', (xls.sheet_names))
            st.sidebar.write(f'Выбран лист: {selected_sheet_names}')

            # Настройка боковой панели
            st.sidebar.subheader('Наименование переменных кластеризации')
            df_initial = pd.read_excel(uploaded_file, sheet_name=selected_sheet_names)
            st.subheader('Фрагмент исходных данные')
            st.write(df_initial.head(5))

            df_initial['Количество наблюдений'] = ''

            # Наименование переменных
            selected_y = st.sidebar.selectbox('Выберите переменную Y', (df_initial.columns))
            selected_x = st.sidebar.selectbox('Выберите переменную X', (df_initial.columns))

            df_y = df_initial[[selected_y]]
            name_y = selected_y + '(Y)'
            df_y = df_y.rename(columns={selected_y: name_y})

            name_x = selected_x + '(X)'
            df_x = df_initial[[selected_x]]
            df_x = df_x.rename(columns={selected_x: name_x})

            name_val_x = 'Количество наблюдений(X)'
            name_val_y = 'Количество наблюдений(Y)'

            st.subheader('Фрагмент выбранных данных кластеризации')

            if (name_y == name_val_y) and (name_x == name_val_x):
                df_y[name_y] = np.arange(len(df_initial))
                df_x[name_x] = np.arange(len(df_initial))
                st.text('Сортирванные исходные данные в порядке убывания')
            elif name_y == name_val_y:
                df_x.sort_values(by=name_x, inplace=True, ascending=False)
                df_x = df_x.reset_index(drop=True)
                df_y[name_y] = np.arange(len(df_initial))
                st.text('Сортирванные исходные данные в порядке убывания')
            elif name_x == name_val_x:
                df_y.sort_values(by=name_y, inplace=True, ascending=False)
                df_y = df_y.reset_index(drop=True)
                df_x[name_x] = np.arange(len(df_initial))
                st.text('Сортирванные исходные данные в порядке убывания')

            # Итоговый датафрейм
            df = pd.concat([df_y,df_x], axis=1)


            # df.sort_values(by=[selected_x1], inplace=True)
            # df_initial['Количество наблюдений'] = np.arange(len(df_initial))

            st.write(df.head(5))

            # Ввод количества кластеров
            st.sidebar.subheader('Ввод количества кластеров')
            n_cluster = int(st.sidebar.number_input('Поле ввода количества кластеров', value=5))
            # sys.exit()






            cluster = create_cluster(df_initial=df, n_cluster=n_cluster)

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

            # fig.add_trace(go.Scatter(x=df_plot.iloc[:,2], y=df_plot.iloc[:,3],
            #                          # line=dict(color='#CF2817'),
            #                          mode='markers',
            #                          marker=dict(color='#ff0000', size=age_fig2, symbol = 'circle'),
            #                          name='Результаты кластеризации'))

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




