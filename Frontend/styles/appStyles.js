import { StyleSheet } from 'react-native';

export const colors = {
  primary: '#ef0000',
  secondary: '#6B7280',
  accent: '#EAF1FF',
  background: '#F7F8FA',
  surface: '#FFFFFF',
  border: '#E5E7EB',
  text: '#172033',
};

const styles = StyleSheet.create({
  app: {
    flex: 1,
    backgroundColor: colors.background,
  },
  screenContainer: {
    flex: 1,
  },
  screen: {
    flex: 1,
    padding: 24,
  },
  homeContent: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  screenTitle: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 24,
  },
  primaryButton: {
    minWidth: 180,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 14,
    backgroundColor: colors.primary,
  },
  primaryButtonDisabled: {
    backgroundColor: colors.secondary,
  },
  primaryButtonLabel: {
    color: colors.surface,
    fontSize: 16,
    fontWeight: '700',
  },
  bottomNav: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: colors.surface,
    borderTopColor: colors.border,
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  navButton: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 4,
    paddingVertical: 12,
    borderRadius: 12,
  },
  navButtonActive: {
    backgroundColor: colors.accent,
  },
  navButtonLocked: {
    opacity: 0.45,
  },
  navLabel: {
    color: colors.secondary,
    fontSize: 14,
    fontWeight: '600',
  },
  navLabelActive: {
    color: colors.primary,
  },
  navLabelLocked: {
    color: colors.secondary,
  },
});

export default styles;
