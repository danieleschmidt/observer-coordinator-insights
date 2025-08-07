"""
Cultural adaptation for neuromorphic clustering visualization and team recommendations
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from . import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass
class CulturalSettings:
    """Cultural settings for a specific locale"""
    locale: str
    color_preferences: Dict[str, str]
    chart_style: str
    team_hierarchy_style: str
    communication_style: str
    decision_making_style: str
    collaboration_preferences: List[str]
    
    
class CulturalAdapter:
    """Adapts clustering visualization and recommendations to cultural preferences"""
    
    def __init__(self):
        self.cultural_settings = self._initialize_cultural_settings()
    
    def _initialize_cultural_settings(self) -> Dict[str, CulturalSettings]:
        """Initialize cultural settings for each supported locale"""
        settings = {}
        
        # English (Western business culture)
        settings['en'] = CulturalSettings(
            locale='en',
            color_preferences={
                'primary': '#2E86AB',      # Professional blue
                'secondary': '#A23B72',    # Confident purple
                'success': '#F18F01',      # Optimistic orange
                'warning': '#C73E1D',      # Alert red
                'team_colors': ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#577590']
            },
            chart_style='minimal',
            team_hierarchy_style='flat',
            communication_style='direct',
            decision_making_style='data_driven',
            collaboration_preferences=['cross_functional', 'agile', 'remote_friendly']
        )
        
        # Spanish (Latin business culture)
        settings['es'] = CulturalSettings(
            locale='es',
            color_preferences={
                'primary': '#C73E1D',      # Warm red
                'secondary': '#F18F01',    # Golden orange
                'success': '#6A994E',      # Natural green
                'warning': '#A23B72',      # Deep purple
                'team_colors': ['#C73E1D', '#F18F01', '#6A994E', '#577590', '#2E86AB']
            },
            chart_style='warm',
            team_hierarchy_style='relationship_based',
            communication_style='personal',
            decision_making_style='consensus_building',
            collaboration_preferences=['relationship_first', 'face_to_face', 'family_oriented']
        )
        
        # French (European corporate culture)
        settings['fr'] = CulturalSettings(
            locale='fr',
            color_preferences={
                'primary': '#1E3A8A',      # Classic blue
                'secondary': '#7C3AED',    # Elegant purple
                'success': '#059669',      # Refined green
                'warning': '#DC2626',      # Classic red
                'team_colors': ['#1E3A8A', '#7C3AED', '#059669', '#B91C1C', '#374151']
            },
            chart_style='elegant',
            team_hierarchy_style='structured',
            communication_style='formal',
            decision_making_style='analytical',
            collaboration_preferences=['expertise_based', 'structured_meetings', 'quality_focused']
        )
        
        # German (Efficiency-focused culture)
        settings['de'] = CulturalSettings(
            locale='de',
            color_preferences={
                'primary': '#374151',      # Professional gray
                'secondary': '#1F2937',    # Dark gray
                'success': '#10B981',      # Efficient green
                'warning': '#F59E0B',      # Warning amber
                'team_colors': ['#374151', '#1F2937', '#10B981', '#3B82F6', '#8B5CF6']
            },
            chart_style='technical',
            team_hierarchy_style='systematic',
            communication_style='precise',
            decision_making_style='thorough_analysis',
            collaboration_preferences=['process_oriented', 'expertise_driven', 'efficiency_focused']
        )
        
        # Japanese (Harmony-oriented culture)
        settings['ja'] = CulturalSettings(
            locale='ja',
            color_preferences={
                'primary': '#475569',      # Subtle slate
                'secondary': '#7C2D12',    # Traditional red
                'success': '#166534',      # Balanced green
                'warning': '#A16207',      # Warm gold
                'team_colors': ['#475569', '#7C2D12', '#166534', '#1E40AF', '#6B21A8']
            },
            chart_style='harmonious',
            team_hierarchy_style='respectful',
            communication_style='indirect',
            decision_making_style='consensus_seeking',
            collaboration_preferences=['harmony_focused', 'respect_hierarchy', 'group_oriented']
        )
        
        # Chinese (Collective achievement culture)
        settings['zh'] = CulturalSettings(
            locale='zh',
            color_preferences={
                'primary': '#DC2626',      # Auspicious red
                'secondary': '#F59E0B',    # Prosperity gold
                'success': '#059669',      # Growth green
                'warning': '#1E40AF',      # Stability blue
                'team_colors': ['#DC2626', '#F59E0B', '#059669', '#1E40AF', '#7C2D12']
            },
            chart_style='symbolic',
            team_hierarchy_style='respectful',
            communication_style='contextual',
            decision_making_style='collective_wisdom',
            collaboration_preferences=['collective_achievement', 'long_term_thinking', 'relationship_building']
        )
        
        return settings
    
    def get_cultural_settings(self, locale: str) -> CulturalSettings:
        """Get cultural settings for a locale"""
        if locale not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported locale {locale}, falling back to English")
            locale = 'en'
        
        return self.cultural_settings.get(locale, self.cultural_settings['en'])
    
    def adapt_color_scheme(self, locale: str) -> Dict[str, str]:
        """Get culturally appropriate color scheme"""
        settings = self.get_cultural_settings(locale)
        return settings.color_preferences
    
    def adapt_chart_visualization(self, locale: str, chart_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt chart visualization to cultural preferences"""
        settings = self.get_cultural_settings(locale)
        colors = settings.color_preferences
        
        adapted_data = chart_data.copy()
        
        # Apply color scheme
        if 'colors' not in adapted_data:
            adapted_data['colors'] = colors['team_colors']
        
        # Apply chart style
        style_config = self._get_chart_style_config(settings.chart_style)
        adapted_data.update(style_config)
        
        return adapted_data
    
    def _get_chart_style_config(self, style: str) -> Dict[str, Any]:
        """Get chart configuration for cultural style"""
        styles = {
            'minimal': {
                'font_family': 'Arial, sans-serif',
                'border_radius': 4,
                'animation_duration': 300,
                'grid_opacity': 0.1
            },
            'warm': {
                'font_family': 'Georgia, serif',
                'border_radius': 8,
                'animation_duration': 500,
                'grid_opacity': 0.2
            },
            'elegant': {
                'font_family': 'Times New Roman, serif',
                'border_radius': 2,
                'animation_duration': 400,
                'grid_opacity': 0.15
            },
            'technical': {
                'font_family': 'Consolas, monospace',
                'border_radius': 0,
                'animation_duration': 200,
                'grid_opacity': 0.3
            },
            'harmonious': {
                'font_family': 'Hiragino Sans, sans-serif',
                'border_radius': 6,
                'animation_duration': 600,
                'grid_opacity': 0.05
            },
            'symbolic': {
                'font_family': 'SimSun, serif',
                'border_radius': 8,
                'animation_duration': 800,
                'grid_opacity': 0.1
            }
        }
        return styles.get(style, styles['minimal'])
    
    def adapt_team_recommendations(self, locale: str, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adapt team recommendations to cultural preferences"""
        settings = self.get_cultural_settings(locale)
        
        adapted_recommendations = []
        
        for rec in recommendations:
            adapted_rec = rec.copy()
            
            # Add cultural context to recommendations
            adapted_rec['cultural_context'] = {
                'hierarchy_style': settings.team_hierarchy_style,
                'communication_style': settings.communication_style,
                'decision_making': settings.decision_making_style,
                'collaboration_preferences': settings.collaboration_preferences
            }
            
            # Adjust recommendation priority based on cultural values
            if settings.decision_making_style == 'consensus_seeking':
                if 'consensus_builder' in rec.get('skills', []):
                    adapted_rec['priority_boost'] = 0.2
            
            elif settings.communication_style == 'direct':
                if 'clear_communicator' in rec.get('skills', []):
                    adapted_rec['priority_boost'] = 0.15
            
            # Add cultural guidance
            adapted_rec['cultural_guidance'] = self._generate_cultural_guidance(locale, rec)
            
            adapted_recommendations.append(adapted_rec)
        
        return adapted_recommendations
    
    def _generate_cultural_guidance(self, locale: str, recommendation: Dict[str, Any]) -> List[str]:
        """Generate culturally appropriate guidance for team formation"""
        settings = self.get_cultural_settings(locale)
        guidance = []
        
        if settings.communication_style == 'direct':
            guidance.append("Encourage open and direct communication")
            guidance.append("Set clear expectations and deadlines")
        
        elif settings.communication_style == 'indirect':
            guidance.append("Allow time for reflection and consensus building")
            guidance.append("Use subtle feedback and respect hierarchies")
        
        if settings.collaboration_preferences:
            for pref in settings.collaboration_preferences:
                if pref == 'relationship_first':
                    guidance.append("Invest time in building personal relationships")
                elif pref == 'process_oriented':
                    guidance.append("Establish clear processes and procedures")
                elif pref == 'harmony_focused':
                    guidance.append("Prioritize team harmony and collective success")
        
        return guidance
    
    def get_localized_team_roles(self, locale: str) -> Dict[str, str]:
        """Get localized team role descriptions"""
        # This would typically load from translation files
        # For now, return basic role mappings
        base_roles = {
            'leader': 'Team Leader',
            'coordinator': 'Coordinator', 
            'analyst': 'Data Analyst',
            'creative': 'Creative Lead',
            'technical': 'Technical Specialist',
            'communicator': 'Communications Lead'
        }
        
        # In a full implementation, this would use the translator
        # to get culturally appropriate role descriptions
        return base_roles
    
    def format_team_metrics(self, locale: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format team metrics according to cultural preferences"""
        settings = self.get_cultural_settings(locale)
        
        formatted_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, float):
                # Format numbers according to locale
                if settings.locale in ['de', 'fr', 'es']:
                    # Use comma as decimal separator
                    formatted_metrics[key] = f"{value:.2f}".replace('.', ',')
                else:
                    formatted_metrics[key] = f"{value:.2f}"
            else:
                formatted_metrics[key] = value
        
        return formatted_metrics