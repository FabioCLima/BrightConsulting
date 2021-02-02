#! importando as bibliotecas necessárias
import os   
from utils import*
import streamlit as st 
import joblib
import base64
import altair as alt
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns 



pd.options.display.max_columns = 200
pd.options.display.float_format = '{:.0f}'.format
pd.set_option('max_rows', 600)
pd.set_option('precision', 1)
from IPython.core.pylabtools import figsize
figsize(12, 8)

plt.style.use('ggplot')  
from PIL import Image



BASE_DIR = os.path.join( os.path.abspath('..') )
IMGS_DIR = os.path.join( BASE_DIR, 'imgs' )
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join( BASE_DIR, 'models' )

def main():
    
    if os.path.isfile( os.path.join( MODELS_DIR, 'xgb_model.pkl' ) ):
        model = pd.read_pickle( os.path.join( MODELS_DIR, 'xgb_model.pkl' ) )
        
    #! Add streamlit title, add descriptions and load an attractive image
 
    st.title("Precificação de veículos utilizando modelos de M.L")
    st.write("Criado por Fabio C. Lima")
    image = Image.open( os.path.join( IMGS_DIR, 'manufacturing_cars.png'))
    image2 = Image.open(os.path.join( IMGS_DIR, 'logomarca_brightconsulting.png'))
    st.image(image, use_column_width=True)
    st.sidebar.image(image2, use_column_width=True)
    st.sidebar.markdown('<br/>'*2, unsafe_allow_html=True)
    
    dados_entrada = st.sidebar.checkbox('Dados de entrada')
    dados_check = st.sidebar.checkbox('Inspeção dos dados de entrada')
    forecast_resultados = st.sidebar.checkbox('Resultados da previsão')
    
    if os.path.isfile( os.path.join( MODELS_DIR, 'xgb_model.pkl' ) ):
        model = pd.read_pickle( os.path.join( MODELS_DIR, 'xgb_model.pkl' ) )
        
    if dados_entrada:
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('### Base de dados que será usada para validação do modelo')
        st.markdown('<br/>'*3, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader('Escolha o arquivo excel para usar o modelo de precificação (.xlsx)')
        if uploaded_file is not None:
            
            @st.cache
            def load_new_data(filename):
                features = [
                    'OEM GROUP',
                    'OEM',
                    'MODEL BRIGHT',
                    'VERSION',
                    'SEGMENT',
                    'SIZE', 
                    'SEGMENT+SIZE',    
                    'EBD (ABS)',
                    'TRACTION CONTROL',
                    'LANE CHANGE BLINK',
                    'FRONT AIRBAGS',
                    'FRONT SIDE\nAIRBAGS',
                    'REAR SIDE\nAIRBAGS',
                    'COURTAIN\nAIRBAGS',
                    'EMERGENCY LIGHTNING',
                    'EMERGENCY SYSTEM (CALL)',
                    'ELECTRONIC STABILITY CONTROL',
                    'DAMPING CONTROL',
                    'NIGHT VISION',
                    'BLIND SPOT MONITORING  ',
                    'FATIGUE DETECTION',
                    'COLLISION MITIGATION BRAKING SYSTEM',
                    'FORWARD COLLISION WARNING',
                    'LANE DEPARTURE',
                    'LANE KEEP ASSIST',
                    'DIGITAL WING MIRRORS',
                    'CROSSWIND STABILIZATION',
                    'ADAPTIVE HIGH BEAM',
                    'REAR TRAFFIC ALERT',
                    'CRUISE CONTROL',
                    'ADAPTIVE CRUISE CONTROL',
                    'CONTROL OF DIRECTION WITH ACTIVE ASSISTANCE (SEMI-AUTONOMOUS)',
                    'TRAFFIC SIGN\nASSIST',
                    'BUCKLE UNLOCK INDICATOR (PASSENGER)',
                    'BUCKLE UNLOCK INDICATOR (PILOT)',
                    'TPMS',
                    'LOW FRICTION \nTIRES',
                    'ECO DRIVING SMART CRUISE CONTROL',
                    'INTELLIGENT ENERGY MANAGEMENT',
                    '48V ECO DRIVE SYSTEM (POWERTRAIN)',
                    '100% ELECTRIC VEHICLE',
                    'GREEN WAVE ASSIST',
                    'ADVANCED SHIFTING',
                    'PLATOONING ',
                    'STOP/START',
                    'LASER LIGHTS',
                    'ACTIVE AERO IMPROVMENT',
                    'GEAR CHANGE  INDICATOR',
                    'AUXILIARY TURN LIGHTS',
                    'HEADLIGHTS WITH DOUBLE REFLECTOR',
                    'HEADLIGHTS WITH TRIPLE REFLECTORS',
                    'HEADLIGHTS WITH MULTI-REFLECTORS',
                    'LED HEADLIGHTS',
                    'HEADLIGHTS WITH BI-XENON DISCHARGE LAMPS',
                    'HEADLIGHTS WITH XENON DISCHARGE LAMPS',
                    'FOG LIGHTS',
                    'LED FOG LIGHTS',
                    'DAYTIME RUNNING LIGHTS',
                    'LED DAYTIME RUNNING LIGHTS',
                    'MAIN HEADLIGHTS WITH INDIVIDUALLY CONTROLLED LEDS (MATRIX)',
                    'MAIN HEADLIGHTS WITH INDIVIDUALLY CONTROLLED LEDS (PIXEL)',
                    'HEADLIGHTS WITH FULL LED SYSTEM',
                    'HEADLIGHTS WITH LED SYSTEM',
                    'REAR FLASHLIGHT WITH DYNAMIC LED FLASHERS',
                    'LED BACKLIGHT',
                    'Headlights with automatic height adjustment',
                    'Headlights with electric height adjustment',
                    'GREEN GLASSES',
                    'DARKENED\nGREEN GLASSES ',
                    'SELECTIVE CATALYST REDUCTION (SCR) (DIESEL)',
                    'CATALYTIC CONVERTER',
                    'DIESEL PARTICLE FILTER REGENERATION SYSTEM (DIESEL)',
                    'VARIABLE COMPRESSOR',
                    'MECHANIC STEERING',
                    'HIDRAULIC STEERING',
                    'ELECTRIC STEERING',
                    'NO AIR CONDITIONING',
                    'AIR CONDITIONING',
                    'AUTOMATIC AIR CONDITIONING',
                    'AUTOMATIC AIR CONDITIONING (2 ZONES)',
                    'AUTOMATIC AIR CONDITIONING (3 ZONES)',
                    'AUTOMATIC AIR CONDITIONING (4 ZONES)',
                    'REVERSE CAMERA',
                    'AUXILIARY PARKING WITH GUIDES',
                    'REAR PARKING SENSORS',
                    'FRONT AND REAR PARKING SENSORS',
                    'PARKING ASSISTANT',
                    'HEADUP DISPLAY',
                    'ENGINE REMOTE FUNCTION',
                    'BUMP/POTHOLE MITIGATION',
                    'HIGHWAY CHAUFFEUR',
                    'AUTOMATED VALET PARKING',
                    'JAM ASSIST',
                    'STEERING WHEEL SWITCHES',
                    'SURROUND VIEW SYSTEM',
                    'GESTURE CONTROL',
                    'BLUETOOTH WITH WIRELESS CHARGING',
                    'HILL ASSIST',
                    'AUTO-HOLD',
                    'ADAPTIVE LOAD CONTROL',
                    'AUXILIARY SYSTEM FOR TRAILLER STABILIZATION',
                    'DOWNHILL ASSIST',
                    'AUTOMATIC HEADLIGHT',
                    'ADAPTIVE HEADLIGHT',
                    'KEYLESS ENTRY',
                    'START-STOP BUTTON',
                    'VEHICLE REMOTE MONITORING',
                    'CONCIERGE SERVICE',
                    'AUTOMATIC TRUNK',
                    'LAUNCH CONTROL',
                    'COLOR LIQUID CRYSTAL DISPLAY ON THE PANEL',
                    'MONOCHROME LCD PANEL',
                    'COLOR LCD PANEL',
                    'ENTIRE DIGITAL AND CONFIGURABLE CLUSTER',
                    "DIGITAL CLUSTER \nPOL ('')",
                    'NO HEAD UNIT (NHU)',
                    'AUDIO SYSTEM (AS)',
                    'DISPLAY AUDIO (DA)',
                    'ANDROID AUTO',
                    'CARPLAY (APPLE)',
                    'MIRROR LINK',
                    'EMBEDDED NAVIGATION (NAV)',
                    "DISPLAY SYSTEM\n POL ('')",
                    ' ON BOARD COMPUTER',
                    'MASS IN RUNNING ORDER (KG)',
                    'DISTANCE BETWEEN AXLES',
                    'AREA (M²)',
                    'CONPET NOVO',
                    'FUEL TYPE',
                    'ENGINE TYPE',
                    'DIRECT INJECTION',
                    'DISPLACEMENT\n(CM³)',
                    'POWER\n(CV)',
                    'TRANSMISSION TYPE',
                    'NUMBER OF GEARS',
                    'CYLINDER NUMBER',
                    'TURBO',
                    'MSRP' 
                    ]
                
                
                data = pd.read_excel(filename, 
                                    sheet_name = 'Planilha1',
                                    header = 1,
                                    usecols = features)
                df = data.copy()
                df.columns = (df.columns.str.lower()
                                    .str.replace(" ","_"))
            
                features = {
                    'displacement\n(cm³)': 'displacement(l)',
                    'power\n(cv)' : 'power(cv)',   
                    'front_side\nairbags' : 'front_side_airbags', 
                    'rear_side\nairbags': 'rear_side_airbags',
                    'courtain\nairbags' : 'courtain_airbags',
                    'blind_spot_monitoring__' : 'blind_spot_monitoring',
                    'traffic_sign\nassist' : 'traffic_sign_assist',
                    'low_friction_\ntires' : "low_friction_tires",
                    'platooning_' : 'platooning',
                    'darkened\ngreen_glasses_' :  'darkened_green_glasses',
                    'bump/pothole_mitigation' : 'bump_pothole_mitigation',
                    "digital_cluster_\npol_('')" : "digital_cluster_pol",
                    'mass_in_running_order_(kg)' : 'mass_in_running_order(kg)',
                    'distance_between_axles' : 'distance_between_axles',
                    'area_(m²)' : 'area(m²)' , 
                    'conpet_novo' : 'conpet_novo', 
                    'fuel_type' : 'fuel_type', 
                    'engine_type' : 'engine_type', 
                    'direct_injection' : 'direct_injection', 
                    'displacement\n(cm³)' : 'displacement(cm³)', 
                    'power\n(cv)' : 'power(cv)', 
                    'transmission_type' : 'transmission_type', 
                    'number_of_gears' : 'number_of_gears', 
                    'cylinder_number' : 'cylinder_number',
                    'gear_change__indicator' : 'gear_change_indicator', 
                    'turbo' : 'turbo',
                    'msrp' : 'prices(R$)' 
                    }
                df.rename(columns = features, inplace = True)
                _ = df.fillna(0, inplace=True)
                _ = df.replace(['N/D', 'X', 'O'], [0,1,1], inplace=True)
                
                features = [
                    'oem_group',
                    'oem',
                    'model_bright',
                    'version',
                    'segment',
                    'size', 
                    'segment+size',
                    'mass_in_running_order(kg)', 
                    'distance_between_axles', 
                    'area(m²)', 
                    'conpet_novo', 
                    'fuel_type', 
                    'engine_type', 
                    'direct_injection', 
                    'displacement(cm³)', 
                    'power(cv)', 
                    'transmission_type', 
                    'number_of_gears', 
                    'cylinder_number', 
                    'turbo',
                    'ebd_(abs)',
                    'traction_control',
                    'lane_change_blink',
                    'front_airbags',
                    'emergency_lightning',
                    'electronic_stability_control',
                    'blind_spot_monitoring',
                    'fatigue_detection',
                    'collision_mitigation_braking_system',
                    'forward_collision_warning',
                    'lane_departure',
                    'lane_keep_assist',
                    'adaptive_high_beam',
                    'cruise_control',
                    'adaptive_cruise_control',
                    'buckle_unlock_indicator_(passenger)',
                    'buckle_unlock_indicator_(pilot)',
                    'tpms',
                    'stop/start',
                    'gear_change_indicator',
                    'led_daytime_running_lights',
                    'led_backlight',
                    'hidraulic_steering',
                    'electric_steering',
                    'air_conditioning',
                    'automatic_air_conditioning',
                    'reverse_camera',
                    'auxiliary_parking_with_guides',
                    'rear_parking_sensors',
                    'front_and_rear_parking_sensors',
                    'parking_assistant',
                    'jam_assist',
                    'bluetooth_with_wireless_charging',
                    'hill_assist',
                    'downhill_assist',
                    'keyless_entry',
                    'start-stop_button',
                    'color_liquid_crystal_display_on_the_panel',
                    'entire_digital_and_configurable_cluster',
                    'audio_system_(as)',
                    'display_audio_(da)',
                    'android_auto',
                    'carplay_(apple)',
                    'embedded_navigation_(nav)',
                    'prices(R$)'   
                    ]
                
                data = df[features].copy()
                features_to_drop = ['oem_group', 'model_bright',\
                'segment+size', 'number_of_gears']
                
                data.drop(features_to_drop, axis=1, inplace=True)
                
    
                df1 = data.drop('prices(R$)', axis = 1).copy()
                df3 = data['prices(R$)'].copy()
                data['total_itens'] = data.iloc[:, 16:59].sum(axis=1)
                df2 = data['total_itens']
                df4 = pd.concat([df1, df2], axis = 1)
                df = pd.concat([df4, df3], axis = 1)
                data = df.copy()
                return data  
            
            df = load_new_data(filename = uploaded_file)  
            
          
            st.markdown('### Visualizando a base de dados de entrada')
            st.markdown('1 - Database inputada ')
            st.markdown('<br/>'*1, unsafe_allow_html=True)
            st.dataframe(df[['oem','version', 'segment', 'prices(R$)']].head(10))
            st.markdown('<br/>'*1, unsafe_allow_html=True)
            
            st.markdown('Número de amostras presentes no arquivo de entrada:')
            st.markdown(df.shape[0])
            st.markdown('Número de features(características do veículo) utilizados no modelo:')
            st.markdown(df.shape[1])    
    
    if dados_check:
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('2 - Features usadas para a construção do modelo')
        variaveis = df.columns.to_list()
        st.write(variaveis)
        
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('3 - Distribuição da variável target no arquivo informado:')
        st.markdown('<br/>'*1, unsafe_allow_html=True)
            
        fig, ax = plt.subplots()
        sns.kdeplot(df['prices(R$)'], shade = True, color = 'b')
        plt.title("Distribuição dos preços no arquivo de entrada")
        st.pyplot(fig)
            
        
    if forecast_resultados:
        st.markdown('<br/>', unsafe_allow_html=True)
        st.markdown('### Resultado da predição usando o modelo de regressão')
        st.markdown('<br/>', unsafe_allow_html=True)
        
    
        version = df['version']
        new_data = df.drop(['prices(R$)', 'version'], axis = 1)
        y_validation = df['prices(R$)']
           
        model = pd.read_pickle(os.path.join( MODELS_DIR, 'xgb_model.pkl' ))
        
        data_scaler = model['model'].named_steps.preprocessor.transform(new_data)
        preds = model['model'].named_steps.xgb_reg.predict(data_scaler)
        residuals = y_validation - preds
        df_results = pd.DataFrame({
                        'oem'                  : new_data['oem'],
                        'segment'              : new_data['segment'],
                        'version'              : version,
                        'preços estimados(R$)' : preds,          # predições/estimativas
                        'preços reais(R$)'     : y_validation,   # preço real
                        'erro(R$)'             : residuals       # diferença
                    }).reset_index(drop = True)
        
        def color_above_12k_red(value):
            if type(value) == type(''):
                return 'color:white'
            else:
                color = 'red' if (value >= 12000 or value <= -12000) else 'white'
                return 'color: {}'.format(color)
        
        st.dataframe(df_results.head(10).sort_values(by='erro(R$)'))
        st.markdown('<br/>'*1, unsafe_allow_html=True)
        
        fig, ax = plt.subplots()
        sns.kdeplot(df['prices(R$)'], color = 'b', shade = True)
        sns.kdeplot(df_results['preços estimados(R$)'], color = 'red', shade = True)
        plt.title("Distribuição dos preços reais x preços previstos")
        st.pyplot(fig)
       
        
                        
if __name__ == '__main__':
    main()
    
    