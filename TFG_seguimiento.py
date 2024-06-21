#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
import numpy as np
from zed_interfaces.msg import ObjectsStamped
from geometry_msgs.msg import Point
from skimage import morphology
from sensor_msgs.msg import Image
from datetime import datetime
import time


class SegNode(Node):
    def __init__(self):
        super().__init__("seg_node")
        self.bridge = CvBridge()
        self.bounding_boxes = {}
        self.person_hog_features = {}
        self.person_centroid_position = {}
        self.bodies_detected = []
        self.tracking_person_id = None
        self.tracking_person_hog = []
        self.tracking_active = False
        self.tracking_lost = False
        
        self.get_logger().info("ROS2 SegNode initialized.")
        # Subscripciones
        self.subscription_image = self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.image_callback, 10)
        self.subscription_depth = self.create_subscription(Image, '/zed/zed_node/depth/depth_registered', self.depth_callback, 10)
        self.subscription_skt = self.create_subscription(ObjectsStamped, '/zed/zed_node/body_trk/skeletons', self.calculate_BoundingBox, 10)
        
        # Publicaciones
        self.pos_publisher = self.create_publisher(Point, "Position_pub", 10)
                
        # Abrir el archivo para guardar datos de posicion
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.position_file = open(f'raul_ws/positions_{timestamp}.csv', 'w')
        self.position_file.write('timestamp,x,y,z\n')
    
      
    def calculate_BoundingBox(self, Bodies):
        
        # Se borran estos datos para tener en cuenta solo las personas presentes en la imagen actual
        self.bodies_detected.clear()
        self.bounding_boxes.clear()
        
        for body in Bodies._objects:
            # Bounding Box de la persona en la imagen
            x = body._bounding_box_2d._corners[0]._kp[0]
            y = body._bounding_box_2d._corners[0]._kp[1]
            w = body._bounding_box_2d._corners[1]._kp[0] - body._bounding_box_2d._corners[0]._kp[0]
            h = body._bounding_box_2d._corners[2]._kp[1] - body._bounding_box_2d._corners[0]._kp[1]

            # Almacenar los cuerpos detectados
            self.bodies_detected.append(body._label)   
            
            # Almacenar las coordenadas de cada BBox
            self.bounding_boxes[body._label] = {'x': x, 'y': y, 'w': w, 'h':h}
            
            # Almacenar la posicion de cada persona
            self.person_centroid_position[body._label] = body._position
        
        self.get_logger().info(f"Bodies detected: {self.bodies_detected}")
        

    def depth_callback(self, msg):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        depth_normalized = depth_image / 20.0
        depth_gray = np.uint8(depth_normalized * 255)
        depth = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)

        self.depth_image = depth    


    def image_callback(self, msg):

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        if not self.tracking_active:
            # Iniciar seguimiento si aún no está activo
            self.start_tracking(image)
        else:
            # Procesar el seguimiento
            self.track_person(image)


    def start_tracking(self, image):
             
        if len(self.bodies_detected) == 0:
            self.get_logger().info("No person found to track.") 

        # El seguimiento se inicia solo si hay una persona detectada en la imagen    
        elif len(self.bodies_detected) == 1:
            person = list(self.bounding_boxes.keys())[0]
            self.tracking_person_id = person
            self.get_logger().info(f"Tracking started for person ID: {self.tracking_person_id}")
            
            self.extract_hog(image)
            
            if self.person_hog_features[person]['n'] == 100:
                self.tracking_active = True
       

    def track_person(self, image):

        self.extract_hog(image)
        
        if self.tracking_person_id in self.bodies_detected:

            self.tracking_person_hog = self.person_hog_features[self.tracking_person_id]['media_hog']
            self.get_logger().info(f"Persona de referencia: {self.tracking_person_id}")

            #Publicación de datos             
            position = self.person_centroid_position[self.tracking_person_id]
            
            position_msg = Point()
            position_msg.x = float(position[0])
            position_msg.y = float(position[1])
            position_msg.z = float(position[2])
            
            self.pos_publisher.publish(position_msg)
            
            # Guardar los datos en el archivo
            now = self.get_clock().now().to_msg()
            timestamp1 = datetime.fromtimestamp(now.sec + now.nanosec * 1e-9).strftime('%Y-%m-%d %H:%M:%S.%f')
            self.position_file.write(f'{timestamp1},{position_msg.x},{position_msg.y},{position_msg.z}\n')
            self.position_file.flush()

        else:
            self.get_logger().info(f"Persona de referencia perdida. Colóquese enfrente del vehículo.")
            self.tracking_lost = True
            self.compare_hogs()
                 
        
    def extract_hog(self, image):
            
        hog = cv2.HOGDescriptor()
        
        # Extraer la región de interés (ROI) de cada persona
        for person_id, bbox in self.bounding_boxes.items():
            
            x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
            
            # Comprobación de que las coordenadas de la ROI estén dentro de los límites de la imagen
            x, y, w, h = max(x, 0), max(y, 0), min(w, image.shape[1] - x), min(h, image.shape[0] - y) 

            # Solo entra a calcular el HOG al principio o cuando se pierde la persona de referencia
            if self.tracking_active == self.tracking_lost:
                
                person_roi = image[y:y+h, x:x+w]
                person_depth_roi = self.depth_image[y:y+h, x:x+w]
                                
                segmented_image_roi = self.calculate_kmeans(person_roi, person_depth_roi)
                
                #Redimensionar la ROI a tamaño estándar
                ROI_size = (128,256)
                resized_person_roi = cv2.resize(segmented_image_roi, ROI_size)                    
                
                gray_roi = cv2.cvtColor(resized_person_roi, cv2.COLOR_BGR2GRAY)
            
                # Calcular el descriptor HOG para la ROI
                fd = hog.compute(gray_roi)
                
                if self.tracking_active == False:
                                                                            
                    if person_id in self.person_hog_features and 'n' in self.person_hog_features[person_id]:                      
                        self.person_hog_features[person_id]['media_hog'] += (fd - self.person_hog_features[person_id]['media_hog']) / (self.person_hog_features[person_id]['n'] + 1)
                        self.person_hog_features[person_id]['n'] += 1
                        self.get_logger().info(f"Media_hog: {self.person_hog_features[person_id]['media_hog']}, n: {self.person_hog_features[person_id]['n']}")
                
                    else:
                        self.person_hog_features[person_id] = {'media_hog': fd, 'n': 1}
                        self.get_logger().info(f"Media_hog: {self.person_hog_features[person_id]['media_hog']}, n: {self.person_hog_features[person_id]['n']}")                   
                
                else: 
                    self.person_hog_features[person_id] = {'media_hog': fd, 'n': 1}             
            
            if person_id == self.tracking_person_id:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
                
                
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, person_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow('Imagen con BBox', image)
            cv2.waitKey(1)  
        
                
    def calculate_kmeans(self, person_roi, person_depth_image):
    
        # Comprobación de que la imagen de profundidad no está vacía
        if person_depth_image.size == 0:
            self.get_logger().info("ROI vacía en imagen de profundidad")
            return None

        data = person_depth_image.reshape((-1, 3))
        data = np.float32(data)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
        K = 5  # Número de clusters
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        segmented_image = res.reshape((person_depth_image.shape))
        
        # Encontrar el cluster más grande
        label = labels.flatten()
        counts = np.bincount(label) 
        largest_cluster = np.argmax(counts) 
        
        # Crear una máscara para el cluster más grande
        mask2 = (labels == largest_cluster).reshape(person_depth_image.shape[:2])
        
        # Dilatación para rellenar huecos
        mask2 = morphology.dilation(mask2, morphology.disk(5))
        
        # Aplicar la máscara a la imagen en color
        segmented_image_mask = np.zeros_like(person_roi)
        segmented_image_mask[mask2] = person_roi[mask2]

        return segmented_image


    def compare_hogs(self):
        
        self.get_logger().info("Comparando HOGs")         

        if not self.bodies_detected:
            self.get_logger().info("No hay descriptores para comparar")
        
        else:
            for body in self.bodies_detected:
                
                descriptor = self.person_hog_features[body]       
        
                #Calcular la distancia Euclidiana
                distancia = np.linalg.norm(self.tracking_person_hog - descriptor['media_hog'])
                        
                # Comprobar si la distancia es menor o igual al umbral
                umbral = 70
                
                if distancia <= umbral:
                    print(f"Referencia detectada. Persona: {body}, Distancia: {distancia}")
                    self.tracking_person_id = body
                    self.tracking_lost = False
                else:
                    print(f"Los descriptores no son similares. Persona: {body},  Distancia: {distancia}")
    
    
    def destroy_node(self):
        # Cerrar el archivo cuando el nodo se destruya
        self.position_file.close()
        super().destroy_node()

        
def main(args=None):
    rclpy.init(args=args)
    node = SegNode()
    
    # Espera inicial de 5 segundos para que se coloque la persona a seguir
    time.sleep(5)
        
    try:
        rclpy.spin(node)
    finally:  
        node.destroy_node()
        rclpy.shutdown()
  
if __name__=='__main__':
    main()